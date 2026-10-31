"""ene - ENhanced Embedded: a container of embedding vectors for video titles (format specification 1.0, SPEC-ene.md).

Modules:
  container   the packet layout (§3): prefix, tables, blocks, trailer, overlay resolution, compaction, the `prep` hash
  api         the public API (§9): open / create / overlay / compact / rewrite, EneFile, Title, Section, selection and search,
              staged edits written as one packet, the local space (§5), a library over several files
  quant       Matryoshka, quantization (fp32 / fp16 / int8 / bin) and the codecs (§4)
  index       the persisted local space (§5.4): the title-part rows of many files stacked, memory-mapped
  probe       ffprobe + hashes for the source file
  media       audio / frame decoding with ffmpeg
  embed       the model of profile ene-1 (§7), the one module that names it: text, image, audio, video, interleaved
  build       profile `ene-1` (§6.1): media -> fp32 master -> .ene
  bench       size / information / speed benchmark (§8.1)
  telemetry   the timed stages (time, CPU, VRAM, RSS, throughput) and the software versions
  analysis    geometry, alignment between sections, queries, cold read, byte profile
  report      telemetry over all the titles of a manifest -> JSON + aggregated report
  serve       the local web GUI (+ webui/)
  cli         python -m ene probe|build|rewrite|edit|overlay|compact|inspect|search|index|bench|telemetry|serve
"""

__version__ = "1.0.0"

from . import index  # noqa: E402,F401
from .api import (DEFAULT_POLICY, DEFAULT_POLICY_NAME, POLICIES, EneFile, Hit, Library, Policy, Section, SectionRef, SectionSpec,  # noqa: E402,F401
                  Space, SpaceHit, Title, Version, VersionSpec, Writer, compact, create, open, overlay, rewrite)
