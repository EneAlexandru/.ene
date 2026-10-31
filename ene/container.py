"""The .ene container, format specification 1.0 (SPEC-ene.md §2–§4): packets made of fixed binary tables and
64-byte-aligned blocks, concatenable end to end, closed by a sha256 trailer; overlay resolution across the packets of
a title (§3.9) and lossless compaction (§3.10). Nothing here knows about models or search: it moves rows and their
provenance between bytes on disk and numpy arrays.

Writing one packet (one title, any number of versions and sections):

    w = PacketWriter(title_id, axis_fps=(30, 1))
    v = w.add_version(version_id, source_hash=sha16, fps=(30, 1), duration=19036, flags=V_REFERENCE | V_TIMELINE)
    w.add_section("synopsis", v, space="eg2-768", modality="text", role="synopsis", lang="eng",
                  vectors=x, quant="fp16", codec="zstd", texts=[synopsis], prep={"profile": "ene-1", "prompt": "title: {title} | text: "})
    w.add_section("frame", v, space="eg2-768", modality="image", role="frame", vectors=f, quant="int8", codec="ene-xdelta",
                  times=frames, prep={...})                       # a `times` column makes a timeline section
    w.set_description({...})                                       # optional JSON, always the last block
    w.write(path, append=False)                                    # append=True adds a packet to an existing file

Reading:

    with EneFile(path) as f:                                       # every packet, grouped by title_id, overlays resolved
        t = f.title()                                              # the only title (or f.titles[title_id])
        for s in t.sections.values():                              # keyed by (version_id, section id)
            s.key, s.vectors(), s.scales(), s.times(), s.axis_times(), s.texts()

Compaction:

    compact(src_path, dst_path)                                    # one packet per title, payloads copied as they are
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
from decimal import Decimal

import numpy as np

from . import quant as Q

MAGIC = b"ENE\x01"
MAGIC_END = b"\x01ENE"
ALIGN = 64
TRAILER_LEN = 32 + 4
ZERO16 = b"\0" * 16

P_ONE_BLOCK = 1                                                   # prefix flag (§3.3, §3.6)
V_REFERENCE, V_TIMELINE = 1, 2                                    # version flags (§2.2)
S_HAS_SCALE, S_HAS_TIME, S_HAS_TEXT, S_ROW_RANGE, S_DELETED, S_TIMELINE = 1, 2, 4, 8, 16, 32   # section flags (§3.5)
MODALITIES = ("text", "image", "video", "audio", "mixed")        # §2.4
MOD_TEXT, MOD_IMAGE, MOD_VIDEO, MOD_AUDIO, MOD_MIXED = range(5)
QUANTS, CODECS = Q.QUANTS, Q.CODECS                               # the numbering of §4.1 / §4.2

READ_WHOLE_BELOW = 4 << 20   # files up to 4 MiB are read() whole; larger ones are memory-mapped (§8.2: a first touch through a
                             # mapping costs ~1 us per 4 KB page on Windows against ~0.1 ms per MB for read())
_O_RD = os.O_RDONLY | getattr(os, "O_BINARY", 0)

# --- the three fixed records (§3.3, §3.4, §3.5); binary ids are void fields so that trailing zero bytes survive ---
PREFIX_DT = np.dtype([("magic", "S4"), ("flags", "<u4"), ("packet_len", "<u8"), ("title_id", "V16"),
                      ("axis_fps_num", "<u4"), ("axis_fps_den", "<u4"), ("n_versions", "<u4"), ("n_sections", "<u4"),
                      ("desc_len", "<u4"), ("desc_raw", "<u4"), ("desc_codec", "u1"), ("trailer_pad", "u1"), ("reserved", "V6")])
VERSION_DT = np.dtype([("version_id", "V16"), ("source_hash", "V16"), ("fps_num", "<u4"), ("fps_den", "<u4"),
                       ("map_a", "<f8"), ("map_b", "<f8"), ("duration", "<i8"), ("rev", "<u4"), ("flags", "<u4"), ("reserved", "V56")])
SECTION_DT = np.dtype([("id", "S32"), ("version", "<u4"), ("lang", "S4"), ("space", "S16"), ("role", "S16"),
                       ("modality", "u1"), ("quant", "u1"), ("codec", "u1"), ("aux_codec", "u1"), ("text_codec", "u1"), ("flags", "u1"),
                       ("reserved0", "<u2"),
                       ("dim", "<u4"), ("dim_real", "<u4"), ("count", "<u4"), ("row_offset", "<u4"), ("rev", "<u4"), ("reserved1", "<u4"),
                       ("vec_off", "<u8"), ("vec_len", "<u8"), ("vec_raw", "<u8"),
                       ("scale_off", "<u8"), ("scale_len", "<u8"), ("scale_raw", "<u8"),
                       ("time_off", "<u8"), ("time_len", "<u8"), ("time_raw", "<u8"),
                       ("text_off", "<u8"), ("text_len", "<u8"), ("text_raw", "<u8"),
                       ("ref", "S32"), ("prep", "<u8"), ("reserved2", "V16")])
assert PREFIX_DT.itemsize == 64 and VERSION_DT.itemsize == 128 and SECTION_DT.itemsize == 256
_BLOCK_FIELDS = (("vectors", "vec_off", "vec_len", "vec_raw"), ("scale", "scale_off", "scale_len", "scale_raw"),
                 ("time", "time_off", "time_len", "time_raw"), ("text", "text_off", "text_len", "text_raw"))
_SF = {name: i for i, name in enumerate(SECTION_DT.names)}   # field positions in a section record turned into a tuple (`tolist`)


def pad64(n: int) -> int:
    return (-n) % ALIGN


def align64(n: int) -> int:
    return n + pad64(n)


def row_bytes(quant: str, dim: int) -> int:
    """Bytes of one row of the `vectors` column (§4.1)."""
    return dim // 8 if quant == "bin" else dim * {"fp32": 4, "fp16": 2, "int8": 1}[quant]


def modality_code(m) -> int:
    return int(m) if not isinstance(m, str) else MODALITIES.index(m)


# --- canonical JSON (RFC 8785) and the `prep` hash (§2.7, §3.5, §3.7) -------------------------------------------------

def _jcs_number(x: float) -> str:
    """A number as ECMAScript's Number.prototype.toString writes it (RFC 8785 §3.2.2.3)."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if not math.isfinite(x):
        raise ValueError("JSON has no NaN or Infinity")
    if x == 0:
        return "0"
    if x.is_integer() and abs(x) < 1e21:
        return str(int(x))
    sign = "-" if x < 0 else ""
    t = Decimal(repr(abs(x))).as_tuple()                           # repr gives the shortest round-trip digits, like ES6
    digits, e = list(t.digits), t.exponent
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop(); e += 1
    k, s = len(digits), "".join(map(str, digits))
    n = k + e                                                      # value = 0.d1…dk × 10^n
    if k <= n <= 21:
        out = s + "0" * (n - k)
    elif 0 < n <= 21:
        out = s[:n] + "." + s[n:]
    elif -6 < n <= 0:
        out = "0." + "0" * (-n) + s
    else:
        e10 = n - 1
        out = s[0] + ("." + s[1:] if k > 1 else "") + "e" + ("+" if e10 > 0 else "-") + str(abs(e10))
    return sign + out


def _jcs(o) -> str:
    if o is None:
        return "null"
    if o is True:
        return "true"
    if o is False:
        return "false"
    if isinstance(o, (int, float, np.integer, np.floating)):
        return _jcs_number(int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else o)
    if isinstance(o, str):
        return json.dumps(o, ensure_ascii=False)                   # the escapes of RFC 8785 §3.2.2.2
    if isinstance(o, (list, tuple)):
        return "[" + ",".join(_jcs(v) for v in o) + "]"
    if isinstance(o, dict):
        items = sorted(((str(k), v) for k, v in o.items()), key=lambda kv: kv[0].encode("utf-16-be"))   # UTF-16 code unit order
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _jcs(v) for k, v in items) + "}"
    raise TypeError(f"not JSON: {type(o).__name__}")


def canonical_json(obj) -> bytes:
    """RFC 8785 canonical JSON: sorted keys, no whitespace, ES6 numbers, minimal escapes; UTF-8."""
    return _jcs(obj).encode("utf-8")


def prep_hash(prep) -> int:
    """The section record's `prep` field: the first 8 bytes (little-endian u64) of the SHA-256 of the canonical JSON of
    the fingerprint object. `None` → 0 (not given); an int is taken as an already computed hash; bytes as canonical JSON."""
    if prep is None:
        return 0
    if isinstance(prep, (int, np.integer)):
        return int(prep)
    data = bytes(prep) if isinstance(prep, (bytes, bytearray, memoryview)) else canonical_json(prep)
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")


# --- writer --------------------------------------------------------------------------------------------------------------

def _void(b: bytes, n: int):
    b = bytes(b)
    if len(b) != n:
        raise ValueError(f"expected {n} bytes, got {len(b)}")
    return np.frombuffer(b, dtype=f"V{n}")[0]


def _sfield(s: str, n: int, what: str) -> bytes:
    b = s.encode("utf-8")
    if len(b) > n:
        raise ValueError(f"{what} longer than {n} bytes: {s!r}")
    return b


class _Blk:
    __slots__ = ("data", "codec", "raw_len")

    def __init__(self, data, codec: str, raw_len: int):
        self.data, self.codec, self.raw_len = bytes(data), codec, raw_len


class PacketWriter:
    """Collects versions, sections and an optional description, then lays out one packet (§3.2)."""

    def __init__(self, title_id: bytes, axis_fps: tuple[int, int]):
        self.title_id = bytes(title_id)
        if len(self.title_id) != 16:
            raise ValueError("title_id must be 16 bytes")
        self.axis_fps = (int(axis_fps[0]), int(axis_fps[1]))
        if self.axis_fps[0] <= 0 or self.axis_fps[1] <= 0:
            raise ValueError("axis_fps must be a positive rational num/den")
        self._versions: list[dict] = []
        self._sections: list[dict] = []
        self._blocks: list[_Blk] = []
        self._desc: _Blk | None = None

    # versions
    def add_version(self, version_id: bytes, *, source_hash: bytes = ZERO16, fps: tuple[int, int] = (0, 0), map_a: float = 1.0,
                    map_b: float = 0.0, duration: int = 0, rev: int = 0, flags: int = 0) -> int:
        """A version record (§2.2, §3.4); returns its index in this packet's table (what `add_section` takes as `version`)."""
        if len(bytes(version_id)) != 16 or len(bytes(source_hash)) != 16:
            raise ValueError("version_id and source_hash must be 16 bytes")
        fps = (int(fps[0]), int(fps[1]))
        if flags & V_TIMELINE:
            if fps[0] <= 0 or fps[1] <= 0:
                raise ValueError("a TIMELINE version needs a frame rate")
        else:
            if fps != (0, 0) or duration != 0:
                raise ValueError("a version without TIMELINE has fps 0/0 and duration 0")
        if flags & V_REFERENCE and (fps != self.axis_fps or map_a != 1.0 or map_b != 0.0 or not flags & V_TIMELINE):
            raise ValueError("the REFERENCE version has the axis frame rate and the identity map")
        self._versions.append(dict(version_id=bytes(version_id), source_hash=bytes(source_hash), fps=fps, map_a=float(map_a),
                                   map_b=float(map_b), duration=int(duration), rev=int(rev), flags=int(flags)))
        return len(self._versions) - 1

    # blocks and sections
    def _add_block(self, data, codec: str, raw_len: int) -> int:
        self._blocks.append(_Blk(data, codec, raw_len))
        return len(self._blocks) - 1

    def _fields(self, id: str, version: int, space: str, modality, role: str, lang: str, quant: str, dim: int, dim_real: int | None,
                count: int, rev: int, ref: str, prep, timeline: bool) -> dict:
        if not 0 <= version < len(self._versions):
            raise ValueError(f"unknown version index {version}")
        if timeline and not self._versions[version]["flags"] & V_TIMELINE:
            raise ValueError("a timeline section needs a version with the TIMELINE flag")
        if quant not in QUANTS:
            raise ValueError(f"unknown quant {quant!r}")
        if count <= 0:
            raise ValueError("a section has at least one row")
        dim_real = int(dim_real) if dim_real is not None else int(dim)
        if dim > dim_real:
            raise ValueError("dim exceeds dim_real")
        return dict(id=_sfield(id, 32, "section id"), version=int(version), lang=_sfield(lang, 4, "lang"), space=_sfield(space, 16, "space"),
                    role=_sfield(role, 16, "role"), modality=modality_code(modality), quant=quant, dim=int(dim), dim_real=dim_real,
                    count=int(count), rev=int(rev), ref=_sfield(ref, 32, "ref"), prep=prep_hash(prep), timeline=bool(timeline),
                    deleted=False, blocks={}, row_offset=None)

    def add_section(self, id: str, version: int, *, space: str, modality, role: str, vectors, quant: str = "fp16", codec: str = "zstd",
                    lang: str = "", dim_real: int | None = None, times=None, texts: list[str] | None = None, rev: int = 0, ref: str = "",
                    prep=None, text_codec: str = "zstd", level: int | None = None) -> None:
        """A section from float32 rows (L2-normalized by the caller). `times` (`i32[count, 2]`, frames of the version) makes it a
        timeline section, sorted by `f0` then `f1` (§2.5; rows, times and texts are permuted together). `prep` is the input
        fingerprint object (§2.7), hashed into the record."""
        x = np.ascontiguousarray(vectors, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("vectors must be a count x dim matrix")
        count, dim = x.shape
        t = None
        if times is not None:
            t = np.ascontiguousarray(times, dtype="<i4")
            if t.shape != (count, 2):
                raise ValueError(f"times must be i32[{count}, 2]")
            if (t[:, 0] < 0).any() or (t[:, 1] < t[:, 0]).any():
                raise ValueError("segments need 0 <= f0 <= f1")
            order = np.lexsort((t[:, 1], t[:, 0]))
            if (order[1:] < order[:-1]).any():
                x, t = x[order], t[order]
                if texts is not None:
                    texts = [texts[i] for i in order]
        if texts is not None and len(texts) != count:
            raise ValueError("one text per row")
        f = self._fields(id, version, space, modality, role, lang, quant, dim, dim_real, count, rev, ref, prep, t is not None)
        qz = Q.quantize(x, quant)
        rb = row_bytes(quant, dim)
        aux = "none" if codec == "none" else "zstd"               # policy `fast`: nothing to decode, not even scale and time
        f["blocks"]["vectors"] = self._add_block(Q.encode(qz.payload, codec, level, rb), codec, len(qz.payload))
        if qz.scale is not None:
            f["blocks"]["scale"] = self._add_block(Q.encode(qz.scale, aux), aux, len(qz.scale))
        if t is not None:
            tb = t.tobytes()
            f["blocks"]["time"] = self._add_block(Q.encode(tb, aux), aux, len(tb))
        if texts is not None:
            xb = "\n".join(s.replace("\n", " ") for s in texts).encode("utf-8")
            f["blocks"]["text"] = self._add_block(Q.encode(xb, text_codec), text_codec, len(xb))
        self._sections.append(f)

    def add_section_encoded(self, fields: dict, vectors: tuple, scale: tuple | None = None, time: tuple | None = None,
                            text: tuple | None = None) -> None:
        """A section whose columns are already encoded: each column is `(encoded_bytes, codec, raw_len)`. `fields` carries
        id, version, space, modality, role, lang, quant, dim, dim_real, count, rev, ref, prep (hash or object), timeline.
        Used by compaction (§3.10) to copy payloads as they are."""
        f = self._fields(fields["id"], fields["version"], fields["space"], fields["modality"], fields["role"], fields.get("lang", ""),
                         fields["quant"], fields["dim"], fields.get("dim_real"), fields["count"], fields.get("rev", 0), fields.get("ref", ""),
                         fields.get("prep"), fields.get("timeline", False))
        for name, col in (("vectors", vectors), ("scale", scale), ("time", time), ("text", text)):
            if col is not None:
                f["blocks"][name] = self._add_block(*col)
        self._sections.append(f)

    def add_shared(self, items: list[dict], quant: str = "int8", codec: str = "none", level: int | None = None) -> None:
        """Several sections as row ranges of ONE `vectors` block (and one `scale` block): the `fast` layout (§3.6, §4.4).
        Each item has the keyword arguments of `add_section` except quant/codec. With `codec = "none"` and nothing else in
        the packet, the prefix gets ONE_BLOCK."""
        prepared = []
        for it in items:
            x = np.ascontiguousarray(it["vectors"], dtype=np.float32)
            count, dim = x.shape
            t = None
            if it.get("times") is not None:
                t = np.ascontiguousarray(it["times"], dtype="<i4").reshape(count, 2)
                order = np.lexsort((t[:, 1], t[:, 0]))
                if (order[1:] < order[:-1]).any():
                    x, t = x[order], t[order]
                    if it.get("texts") is not None:
                        it = dict(it, texts=[it["texts"][i] for i in order])
            f = self._fields(it["id"], it["version"], it["space"], it["modality"], it["role"], it.get("lang", ""), quant, dim, it.get("dim_real"),
                             count, it.get("rev", 0), it.get("ref", ""), it.get("prep"), t is not None)
            prepared.append((f, Q.quantize(x, quant), t, it.get("texts")))
        self._add_shared(prepared, quant, codec, level)

    def add_shared_encoded(self, items: list[tuple], quant: str, codec: str = "none") -> None:
        """`add_shared` for already quantized rows: items are `(fields, raw_vectors_bytes, raw_scale_bytes | None,
        time_column | None, text_column | None)` with the columns as in `add_section_encoded`."""
        prepared = []
        for fields, vb, sb, time, text in items:
            f = self._fields(fields["id"], fields["version"], fields["space"], fields["modality"], fields["role"], fields.get("lang", ""),
                             quant, fields["dim"], fields.get("dim_real"), fields["count"], fields.get("rev", 0), fields.get("ref", ""),
                             fields.get("prep"), fields.get("timeline", False))
            prepared.append((f, Q.Quantized(quant, bytes(vb), None if sb is None else bytes(sb), fields["count"], fields["dim"]), time, text))
        self._add_shared(prepared, quant, codec, None)

    def _add_shared(self, prepared: list, quant: str, codec: str, level) -> None:
        if not prepared:
            return
        dims = {qz.dim for _, qz, _, _ in prepared}
        if len(dims) != 1:
            raise ValueError("shared sections must have one dim")
        prepared.sort(key=lambda p: (p[0]["timeline"], p[0]["count"], p[0]["id"]))   # the table order of §2.5 = row order
        dim = dims.pop()
        rb = row_bytes(quant, dim)
        payload = b"".join(qz.payload for _, qz, _, _ in prepared)
        vb = self._add_block(Q.encode(payload, codec, level, rb), codec, len(payload))
        aux = "none" if codec == "none" else "zstd"
        sb = None
        if prepared[0][1].scale is not None:
            sc = b"".join(qz.scale for _, qz, _, _ in prepared)
            sb = self._add_block(Q.encode(sc, aux), aux, len(sc))
        off = 0
        for f, qz, t, texts in prepared:
            f["blocks"]["vectors"] = vb
            f["row_offset"] = off
            if sb is not None:
                f["blocks"]["scale"] = sb
            if t is not None:
                if isinstance(t, tuple):                           # an encoded column
                    f["blocks"]["time"] = self._add_block(*t)
                else:
                    tb = np.ascontiguousarray(t, dtype="<i4").tobytes()
                    f["blocks"]["time"] = self._add_block(Q.encode(tb, aux), aux, len(tb))
            if texts is not None:
                if isinstance(texts, tuple):
                    f["blocks"]["text"] = self._add_block(*texts)
                else:
                    xb = "\n".join(s.replace("\n", " ") for s in texts).encode("utf-8")
                    f["blocks"]["text"] = self._add_block(Q.encode(xb, "zstd"), "zstd", len(xb))
            self._sections.append(f)
            off += qz.count

    def add_block(self, data, codec: str = "none", level: int | None = None) -> int:
        """A block that no section names: addressed through the description only (the persisted local space of §5.4 keeps
        its `origin` columns and its file table this way). Returns the block index; `block_offsets()` gives its packet
        offset once the sections are in."""
        enc = Q.encode(bytes(data), codec, level)
        return self._add_block(enc, codec, len(bytes(data)))

    def add_tombstone(self, id: str, version: int, rev: int = 0) -> None:
        """A DELETED record: removes `(version_id, id)` from the title when read after the section it names (§3.9)."""
        if not 0 <= version < len(self._versions):
            raise ValueError(f"unknown version index {version}")
        self._sections.append(dict(id=_sfield(id, 32, "section id"), version=int(version), rev=int(rev), deleted=True, timeline=False,
                                   count=0, blocks={}, row_offset=None))

    def set_description(self, desc: dict, codec: str = "zstd") -> None:
        raw = json.dumps(desc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._desc = _Blk(Q.encode(raw, codec), codec, len(raw))

    # layout
    def _layout(self):
        """Table order (§2.5), the end of the tables and the packet offset of every block: the columns of the sections in
        table order, then the blocks no section names (`add_block`), each 64-byte aligned; the description comes last."""
        secs = sorted(self._sections, key=lambda s: (s["timeline"], s["count"], s["id"]))     # §2.5: title part first, ascending
        nv, ns = len(self._versions), len(secs)
        tables_end = 64 + 128 * nv + 256 * ns
        pos = align64(tables_end)
        offsets: dict[int, int] = {}
        order: list[int] = []
        for s in secs:
            for name, *_ in _BLOCK_FIELDS:
                bi = s["blocks"].get(name)
                if bi is not None and bi not in offsets:
                    offsets[bi] = pos
                    order.append(bi)
                    pos += align64(len(self._blocks[bi].data))
        for bi in range(len(self._blocks)):
            if bi not in offsets:
                offsets[bi] = pos
                order.append(bi)
                pos += align64(len(self._blocks[bi].data))
        return secs, nv, ns, tables_end, pos, offsets, order

    def block_offsets(self) -> dict[int, int]:
        """Packet offset of every block as `to_bytes` will place it (the description, always last, moves none of them)."""
        return self._layout()[5]

    def to_bytes(self) -> bytes:
        secs, nv, ns, tables_end, pos, offsets, order = self._layout()
        desc_off = None
        if self._desc is not None:
            desc_off = pos
            pos += align64(len(self._desc.data))
        trailer_off = pos
        tpad = pad64(trailer_off + TRAILER_LEN)
        packet_len = trailer_off + TRAILER_LEN + tpad

        vt = np.zeros(nv, VERSION_DT)
        for i, v in enumerate(self._versions):
            vt["version_id"][i] = _void(v["version_id"], 16)
            vt["source_hash"][i] = _void(v["source_hash"], 16)
            vt["fps_num"][i], vt["fps_den"][i] = v["fps"]
            vt["map_a"][i], vt["map_b"][i], vt["duration"][i], vt["rev"][i], vt["flags"][i] = v["map_a"], v["map_b"], v["duration"], v["rev"], v["flags"]
        st = np.zeros(ns, SECTION_DT)
        for i, s in enumerate(secs):
            st["id"][i], st["version"][i], st["rev"][i] = s["id"], s["version"], s["rev"]
            if s["deleted"]:
                st["flags"][i] = S_DELETED                         # a tombstone carries only id, version and rev (§3.5)
                continue
            st["lang"][i], st["space"][i], st["role"][i], st["modality"][i] = s["lang"], s["space"], s["role"], s["modality"]
            st["quant"][i] = QUANTS.index(s["quant"])
            st["dim"][i], st["dim_real"][i], st["count"][i] = s["dim"], s["dim_real"], s["count"]
            st["ref"][i], st["prep"][i] = s["ref"], s["prep"]
            flags = S_TIMELINE if s["timeline"] else 0
            if s["row_offset"] is not None:
                flags |= S_ROW_RANGE
                st["row_offset"][i] = s["row_offset"]
            for name, f_off, f_len, f_raw in _BLOCK_FIELDS:
                bi = s["blocks"].get(name)
                if bi is None:
                    continue
                b = self._blocks[bi]
                st[f_off][i], st[f_len][i], st[f_raw][i] = offsets[bi], len(b.data), b.raw_len
                if name == "vectors":
                    st["codec"][i] = CODECS.index(b.codec)
                elif name == "text":
                    flags |= S_HAS_TEXT
                    st["text_codec"][i] = CODECS.index(b.codec)
                else:
                    flags |= S_HAS_SCALE if name == "scale" else S_HAS_TIME
                    st["aux_codec"][i] = CODECS.index(b.codec)
            st["flags"][i] = flags
        live = [s for s in secs if not s["deleted"]]
        one = bool(live) and len(live) == ns and all(s["row_offset"] is not None for s in live) \
            and len({s["blocks"]["vectors"] for s in live}) == 1 and self._blocks[live[0]["blocks"]["vectors"]].codec == "none" \
            and all(live[i]["row_offset"] <= live[i + 1]["row_offset"] for i in range(len(live) - 1))

        p = np.zeros(1, PREFIX_DT)
        p["magic"], p["flags"], p["packet_len"] = MAGIC, P_ONE_BLOCK if one else 0, packet_len
        p["title_id"][0] = _void(self.title_id, 16)
        p["axis_fps_num"], p["axis_fps_den"], p["n_versions"], p["n_sections"] = self.axis_fps[0], self.axis_fps[1], nv, ns
        if self._desc is not None:
            p["desc_len"], p["desc_raw"], p["desc_codec"] = len(self._desc.data), self._desc.raw_len, CODECS.index(self._desc.codec)
        p["trailer_pad"] = tpad

        body = bytearray(packet_len)
        body[0:64] = p.tobytes()
        body[64:64 + 128 * nv] = vt.tobytes()
        body[64 + 128 * nv:tables_end] = st.tobytes()
        for bi in order:
            d = self._blocks[bi].data
            body[offsets[bi]:offsets[bi] + len(d)] = d
        if self._desc is not None:
            body[desc_off:desc_off + len(self._desc.data)] = self._desc.data
        body[trailer_off:trailer_off + 32] = hashlib.sha256(memoryview(body)[:trailer_off]).digest()
        body[trailer_off + 32:trailer_off + 36] = MAGIC_END
        return bytes(body)

    def write(self, path: str, append: bool = False) -> int:
        data = self.to_bytes()
        with open(path, "ab" if append else "wb") as f:
            f.write(data)
        return len(data)


# --- reader --------------------------------------------------------------------------------------------------------------

class Packet:
    """One packet of a file: its prefix, both tables (copied out of the buffer) and access to its blocks."""
    __slots__ = ("file", "index", "offset", "length", "flags", "title_id", "axis_fps", "versions", "sections",
                 "desc_len", "desc_raw", "desc_codec", "trailer_pad", "hash_ok", "_vids")

    def __init__(self, file: "EneFile", index: int, offset: int):
        mv = file._mv
        if offset + 64 > len(mv):
            raise ValueError(f"truncated prefix at offset {offset}")
        p = np.frombuffer(bytes(mv[offset:offset + 64]), PREFIX_DT)[0]
        if bytes(p["magic"]) != MAGIC:
            raise ValueError(f"invalid magic at offset {offset}")
        self.file, self.index, self.offset = file, index, offset
        self.length = int(p["packet_len"])
        if self.length % ALIGN or offset + self.length > len(mv):
            raise ValueError(f"truncated packet at offset {offset}: {self.length} bytes declared, {len(mv) - offset} present")
        self.flags = int(p["flags"])
        self.title_id = bytes(p["title_id"])
        self.axis_fps = (int(p["axis_fps_num"]), int(p["axis_fps_den"]))
        nv, ns = int(p["n_versions"]), int(p["n_sections"])
        a = offset + 64
        self.versions = np.frombuffer(bytes(mv[a:a + 128 * nv]), VERSION_DT)
        a += 128 * nv
        self.sections = np.frombuffer(bytes(mv[a:a + 256 * ns]), SECTION_DT)
        self.desc_len, self.desc_raw, self.desc_codec = int(p["desc_len"]), int(p["desc_raw"]), int(p["desc_codec"])
        self.trailer_pad = int(p["trailer_pad"])
        t = self.trailer_offset
        if bytes(mv[t + 32:t + 36]) != MAGIC_END:
            raise ValueError(f"invalid trailer in packet {index}")
        self.hash_ok: bool | None = None
        self._vids: list[bytes] | None = None

    @property
    def trailer_offset(self) -> int:
        return self.offset + self.length - self.trailer_pad - TRAILER_LEN

    @property
    def one_block(self) -> bool:
        return bool(self.flags & P_ONE_BLOCK)

    def verify(self) -> bool:
        """sha256 of the packet up to the trailer against the stored digest (§3.8); ~1 ms per MB."""
        mv, t = self.file._mv, self.trailer_offset
        self.hash_ok = hashlib.sha256(mv[self.offset:t]).digest() == bytes(mv[t:t + 32])
        return self.hash_ok

    def version_id(self, index: int) -> bytes:
        ids = self._vids
        if ids is None:
            ids = self._vids = [bytes(v) for v in self.versions["version_id"].tolist()]
        return ids[index]

    def description(self) -> dict | None:
        """The description block, decoded on demand (§3.7); None when the packet has none."""
        if self.desc_len == 0:
            return None
        off = self.trailer_offset - align64(self.desc_len)
        raw = Q.decode(self.file._mv[off:off + self.desc_len], CODECS[self.desc_codec], self.desc_raw)
        return json.loads(bytes(raw))

    def block(self, off: int, enc_len: int, raw_len: int, codec: int, rb: int | None = None):
        """A decoded column: a memoryview on the file bytes for codec `none` (zero copy; keep the file open while using it),
        bytes or a memoryview over a fresh array otherwise."""
        raw = self.file._mv[self.offset + off:self.offset + off + enc_len]
        if codec == 0:
            return raw
        return Q.decode(raw, CODECS[codec], raw_len, rb)

    def block_encoded(self, off: int, enc_len: int) -> memoryview:
        return self.file._mv[self.offset + off:self.offset + off + enc_len]


class Version:
    __slots__ = ("packet", "id", "source_hash", "fps", "map_a", "map_b", "duration", "rev", "flags")

    def __init__(self, packet: Packet, rec):
        self.packet = packet
        self.id, self.source_hash = bytes(rec["version_id"]), bytes(rec["source_hash"])
        self.fps = (int(rec["fps_num"]), int(rec["fps_den"]))
        self.map_a, self.map_b = float(rec["map_a"]), float(rec["map_b"])
        self.duration, self.rev, self.flags = int(rec["duration"]), int(rec["rev"]), int(rec["flags"])

    @property
    def reference(self) -> bool:
        return bool(self.flags & V_REFERENCE)

    @property
    def timeline(self) -> bool:
        return bool(self.flags & V_TIMELINE)

    def to_axis(self, segments: np.ndarray) -> np.ndarray:
        """Version frames `[f0, f1)` to axis frames: `floor(a·f0 + b)`, `ceil(a·f1 + b)` (§2.8); int64[..., 2]."""
        s = np.asarray(segments, dtype=np.float64)
        out = np.empty(s.shape, dtype=np.int64)
        out[..., 0] = np.floor(self.map_a * s[..., 0] + self.map_b)
        out[..., 1] = np.ceil(self.map_a * s[..., 1] + self.map_b)
        return out

    def __repr__(self) -> str:
        return f"Version({self.id.hex()[:8]}…, fps {self.fps[0]}/{self.fps[1]}, map {self.map_a}·f+{self.map_b}, rev {self.rev}, flags {self.flags})"


class Section:
    """One resolved section: the record's fields as attributes, the columns on demand."""
    __slots__ = ("title", "packet", "rec", "version_id", "id", "space", "role", "lang", "modality", "quant", "codec", "dim", "dim_real",
                 "count", "rev", "flags", "ref", "prep", "row_offset")

    def __init__(self, title: "Title", packet: Packet, rec, tup=None):
        """`rec`: the numpy record (kept for the column offsets); `tup`: the same record as a tuple of Python scalars (one
        `tolist()` per table, §8.2: ~5 us for a whole table against ~1 us per numpy scalar read)."""
        self.title, self.packet, self.rec = title, packet, rec
        t = tup if tup is not None else rec.tolist()
        f = _SF
        self.version_id = packet.version_id(t[f["version"]])
        self.id = t[f["id"]].decode("utf-8")
        self.space, self.role, self.lang = t[f["space"]].decode("utf-8"), t[f["role"]].decode("utf-8"), t[f["lang"]].decode("utf-8")
        self.modality, self.flags = t[f["modality"]], t[f["flags"]]
        self.quant, self.codec = QUANTS[t[f["quant"]]], CODECS[t[f["codec"]]]
        self.dim, self.dim_real, self.count, self.rev = t[f["dim"]], t[f["dim_real"]], t[f["count"]], t[f["rev"]]
        self.ref = t[f["ref"]].decode("utf-8")
        self.prep = t[f["prep"]]
        self.row_offset = t[f["row_offset"]] if self.flags & S_ROW_RANGE else None

    @property
    def version(self) -> Version:
        return self.title.versions[self.version_id]

    @property
    def timeline(self) -> bool:
        return bool(self.flags & S_TIMELINE)

    @property
    def modality_name(self) -> str:
        return MODALITIES[self.modality] if self.modality < len(MODALITIES) else str(self.modality)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.space, self.quant, self.dim)

    @property
    def row_bytes(self) -> int:
        return row_bytes(self.quant, self.dim)

    def _col(self, name: str):
        f_off, f_len, f_raw = {n: (o, l, r) for n, o, l, r in _BLOCK_FIELDS}[name]
        codec = int(self.rec["codec"]) if name == "vectors" else int(self.rec["text_codec"]) if name == "text" else int(self.rec["aux_codec"])
        return self.packet.block(int(self.rec[f_off]), int(self.rec[f_len]), int(self.rec[f_raw]), codec, self.row_bytes if name == "vectors" else None)

    def vector_bytes(self):
        """The rows of the `vectors` column (the section's slice when the block is shared, §3.6)."""
        b = self._col("vectors")
        if self.row_offset is not None:
            rb = self.row_bytes
            b = b[self.row_offset * rb:(self.row_offset + self.count) * rb]
        return b

    def scale_bytes(self):
        if not self.flags & S_HAS_SCALE:
            return None
        b = self._col("scale")
        if self.row_offset is not None:
            b = b[self.row_offset * 4:(self.row_offset + self.count) * 4]
        return b

    def vectors(self, raw: bool = False) -> np.ndarray:
        """float32 rows (dequantized), or with `raw=True` a zero-copy view in the stored element type."""
        payload = self.vector_bytes()
        if raw:
            return Q.raw_view(payload, self.quant, self.count, self.dim)
        v = Q.dequantize(payload, self.quant, self.count, self.dim, self.scale_bytes())
        return v.copy() if isinstance(payload, memoryview) else v

    def scales(self) -> np.ndarray | None:
        b = self.scale_bytes()
        return None if b is None else np.frombuffer(b, dtype="<f4")

    def times(self) -> np.ndarray | None:
        """`i32[count, 2]` segments in the version's frames; None in the title part."""
        if not self.flags & S_HAS_TIME:
            return None
        return np.frombuffer(self._col("time"), dtype="<i4").reshape(self.count, 2)

    def axis_times(self) -> np.ndarray | None:
        """The segments on the title's frame axis (§2.8), through the version's current map."""
        t = self.times()
        return None if t is None else self.version.to_axis(t)

    def texts(self) -> list[str] | None:
        if not self.flags & S_HAS_TEXT:
            return None
        return bytes(self._col("text")).decode("utf-8").split("\n")

    def encoded(self, name: str) -> tuple | None:
        """`(encoded_bytes, codec, raw_len)` of a column as stored, for copying without decoding; None if absent."""
        f_off, f_len, f_raw = {n: (o, l, r) for n, o, l, r in _BLOCK_FIELDS}[name]
        has = {"vectors": True, "scale": self.flags & S_HAS_SCALE, "time": self.flags & S_HAS_TIME, "text": self.flags & S_HAS_TEXT}[name]
        if not has:
            return None
        codec = self.codec if name == "vectors" else CODECS[int(self.rec["text_codec"])] if name == "text" else CODECS[int(self.rec["aux_codec"])]
        return (self.packet.block_encoded(int(self.rec[f_off]), int(self.rec[f_len])), codec, int(self.rec[f_raw]))

    def fields(self) -> dict:
        """The record's identity fields, in the shape `PacketWriter.add_section_encoded` takes (without `version`)."""
        return dict(id=self.id, space=self.space, modality=self.modality, role=self.role, lang=self.lang, quant=self.quant, dim=self.dim,
                    dim_real=self.dim_real, count=self.count, rev=self.rev, ref=self.ref, prep=self.prep, timeline=self.timeline)

    def __repr__(self) -> str:
        return f"Section({self.id!r}, v {self.version_id.hex()[:8]}…, {self.modality_name}/{self.role}, {self.count}×{self.dim} {self.quant}+{self.codec}, rev {self.rev})"


def _merge(a: dict, b: dict) -> dict:
    """Description overlay (§3.9): key by key, at every level, the later packet winning on every key it carries."""
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Title:
    """Everything the file says about one `title_id`, overlays resolved (§3.9)."""

    def __init__(self, title_id: bytes, axis_fps: tuple[int, int]):
        self.title_id, self.axis_fps = title_id, axis_fps
        self.packets: list[Packet] = []
        self.versions: dict[bytes, Version] = {}
        self.sections: dict[tuple[bytes, str], Section] = {}
        self._desc: dict | None = None

    @property
    def description(self) -> dict:
        if self._desc is None:
            d: dict = {}
            for pk in self.packets:
                x = pk.description()
                if x:
                    d = _merge(d, x)
            self._desc = d
        return self._desc

    def section(self, id: str, version: bytes | None = None) -> Section:
        """A section by id, and by version when the id exists in several versions."""
        if version is not None:
            return self.sections[(version, id)]
        hits = [s for (vid, sid), s in self.sections.items() if sid == id]
        if len(hits) != 1:
            raise KeyError(f"{len(hits)} sections named {id!r}; give the version")
        return hits[0]

    def describe(self) -> str:
        lines = [f"title {self.title_id.hex()} axis {self.axis_fps[0]}/{self.axis_fps[1]} fps, {len(self.packets)} packet(s), "
                 f"{len(self.versions)} version(s), {len(self.sections)} section(s)"]
        for v in self.versions.values():
            lines.append(f"  {v!r}")
        lines.append(f"  {'section':<24} {'version':<9} {'part':<8} {'mod':<5} {'role':<10} {'lang':<4} {'rows':>6} {'dim':>4} {'quant':<5} {'codec':<10} {'rev':>3}")
        for (vid, sid), s in self.sections.items():
            lines.append(f"  {sid:<24} {vid.hex()[:8]:<9} {'timeline' if s.timeline else 'title':<8} {s.modality_name[:5]:<5} {s.role:<10} {s.lang:<4} "
                         f"{s.count:>6} {s.dim:>4} {s.quant:<5} {s.codec:<10} {s.rev:>3}")
        return "\n".join(lines)


class EneFile:
    """A file: its packets in order and, resolved on first access, its titles."""

    def __init__(self, path: str, verify: bool = True):
        self.path = path
        fd = os.open(path, _O_RD)
        try:
            size = os.lseek(fd, 0, 2)
            if size <= READ_WHOLE_BELOW:
                os.lseek(fd, 0, 0)
                data = os.read(fd, size)
                while len(data) < size:
                    more = os.read(fd, size - len(data))
                    if not more:
                        break
                    data += more
                self._buf = data
            else:
                self._buf = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)
        self._mv = memoryview(self._buf)
        self.packets: list[Packet] = []
        pos, n = 0, len(self._mv)
        while pos < n:
            pk = Packet(self, len(self.packets), pos)
            if verify and not pk.verify():
                raise ValueError(f"sha256 mismatch in packet {pk.index} of {path}")
            self.packets.append(pk)
            pos += pk.length
        self._titles: dict[bytes, Title] | None = None

    @property
    def mapped(self) -> bool:
        return isinstance(self._buf, mmap.mmap)

    def verify(self) -> bool:
        return all(pk.verify() for pk in self.packets)

    @property
    def titles(self) -> dict[bytes, Title]:
        if self._titles is None:
            titles: dict[bytes, Title] = {}
            for pk in self.packets:
                t = titles.get(pk.title_id)
                if t is None:
                    t = titles[pk.title_id] = Title(pk.title_id, pk.axis_fps)
                elif pk.axis_fps != t.axis_fps:
                    raise ValueError(f"packet {pk.index} declares axis {pk.axis_fps} for a title whose axis is {t.axis_fps}")
                t.packets.append(pk)
                for i in range(len(pk.versions)):
                    v = Version(pk, pk.versions[i])
                    old = t.versions.get(v.id)
                    if old is None or v.rev >= old.rev:                # higher rev replaces; equal rev: the later packet
                        t.versions[v.id] = v
                f_id, f_ver, f_rev, f_flags = _SF["id"], _SF["version"], _SF["rev"], _SF["flags"]
                nv = len(pk.versions)
                for i, tup in enumerate(pk.sections.tolist()):           # one tolist(): the whole table as Python scalars
                    vi = tup[f_ver]
                    if vi >= nv:
                        raise ValueError(f"section {tup[f_id]!r} of packet {pk.index} names version {vi} of {nv}")
                    key = (pk.version_id(vi), tup[f_id].decode("utf-8"))
                    rev, old = tup[f_rev], t.sections.get(key)
                    if tup[f_flags] & S_DELETED:
                        if old is None or rev >= old.rev:               # a tombstone older than the section it names is stale
                            t.sections.pop(key, None)
                    elif old is None or rev >= old.rev:
                        if key[0] not in t.versions:
                            raise ValueError(f"section {key[1]!r} belongs to an unknown version")
                        t.sections[key] = Section(t, pk, pk.sections[i], tup)
            self._titles = titles
        return self._titles

    def title(self, title_id: bytes | None = None) -> Title:
        if title_id is not None:
            return self.titles[title_id]
        if len(self.titles) != 1:
            raise KeyError(f"{len(self.titles)} titles in {self.path}; give the title_id")
        return next(iter(self.titles.values()))

    def close(self):
        if isinstance(self._buf, mmap.mmap):
            self._mv.release()
            try:
                self._buf.close()
            except BufferError:
                pass                                               # raw views are still alive; the mapping closes with them
        self._buf, self._mv = b"", memoryview(b"")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# --- compaction (§3.10) --------------------------------------------------------------------------------------------------

def compact_bytes(f: EneFile) -> bytes:
    """One packet per title with the overlays resolved, sections reordered (§2.5) and payloads copied as they are. Titles
    whose surviving sections are all raw (`codec = none`) of one quant and width come out as one shared block (ONE_BLOCK)."""
    out = []
    for tid, t in f.titles.items():
        w = PacketWriter(tid, t.axis_fps)
        vidx = {vid: w.add_version(vid, source_hash=v.source_hash, fps=v.fps, map_a=v.map_a, map_b=v.map_b, duration=v.duration,
                                   rev=v.rev, flags=v.flags) for vid, v in t.versions.items()}
        secs = list(t.sections.values())
        if secs and all(s.codec == "none" for s in secs) and len({(s.quant, s.dim) for s in secs}) == 1:
            w.add_shared_encoded([(dict(s.fields(), version=vidx[s.version_id]), s.vector_bytes(), s.scale_bytes(), s.encoded("time"), s.encoded("text"))
                                  for s in secs], secs[0].quant, "none")
        else:
            for s in secs:
                if s.row_offset is None:
                    vec, sc = s.encoded("vectors"), s.encoded("scale")
                else:                                              # a row range of a shared block: its slice, re-encoded losslessly
                    vb = bytes(s.vector_bytes())
                    vec = (Q.encode(vb, s.codec, row_bytes=s.row_bytes), s.codec, len(vb))
                    sb = s.scale_bytes()
                    sc = None if sb is None else (Q.encode(bytes(sb), "none" if s.codec == "none" else "zstd"), "none" if s.codec == "none" else "zstd", len(sb))
                w.add_section_encoded(dict(s.fields(), version=vidx[s.version_id]), vec, sc, s.encoded("time"), s.encoded("text"))
        if t.description:
            w.set_description(t.description)
        out.append(w.to_bytes())
    return b"".join(out)


def compact(src: str, dst: str) -> int:
    with EneFile(src) as f:
        data = compact_bytes(f)
    with open(dst, "wb") as o:
        o.write(data)
    return len(data)
