"""Telemetry (SPEC §13): the runtime environment and the timed stages of the pipeline.

    tel = Telemetry()
    with tel.stage("L2.audio.30s", items=n, media_s=sec) as rec:
        ...                      # rec is a dict; fields can be added from inside the block
    tel.to_dict()                # {"host": ..., "stages": [...], "total_wall_s": ...}

Each stage: wall time, process CPU time, peak VRAM allocated during the stage (torch), VRAM allocated after,
RSS after and its delta, time offset since start; derived: items/s and the real-time factor (seconds of media per
second of compute). Stages that run in parallel (media decoding) are marked `parallel: true`, because the process
CPU time cannot be attributed to them individually.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

GIB = 2**30


# --- environment --------------------------------------------------------------------

def rss_bytes() -> int | None:
    """Resident memory of the current process (Windows: psapi; otherwise /proc/self/statm)."""
    try:
        if sys.platform == "win32":
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.GetCurrentProcess.restype = ctypes.c_void_p                 # 64-bit HANDLE, otherwise ctypes truncates it to int
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_uint32]
            pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return int(pmc.WorkingSetSize)
            return None
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001
        return None


def ram_total_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                            ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]
            ms = MS(); ms.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(ms.ullTotalPhys)
            return None
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:  # noqa: BLE001
        return None


def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return " ".join(winreg.QueryValueEx(k, "ProcessorNameString")[0].split())
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "?"


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def gpu_info() -> dict:
    d: dict = {}
    try:
        import torch
        d["cuda_available"] = torch.cuda.is_available()
        d["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            d |= {"name": p.name, "vram_gib": round(p.total_memory / GIB, 2), "capability": f"{p.major}.{p.minor}", "sms": p.multi_processor_count,
                  "bf16": torch.cuda.is_bf16_supported()}
    except Exception as e:  # noqa: BLE001
        d["error"] = str(e)
    smi = _run(["nvidia-smi", "--query-gpu=driver_version,memory.used,memory.total,clocks.max.sm,power.limit", "--format=csv,noheader,nounits"])
    if smi:
        parts = [x.strip() for x in smi.splitlines()[0].split(",")]
        if len(parts) >= 5:
            d |= {"driver": parts[0], "vram_used_mib_at_start": int(float(parts[1])), "clock_sm_max_mhz": int(float(parts[3])), "power_limit_w": float(parts[4])}
    return d


def lib_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "sentence_transformers", "numpy", "zstandard", "PIL", "av", "fastapi"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
    ff = _run(["ffmpeg", "-version"])
    out["ffmpeg"] = ff.splitlines()[0].split("Copyright")[0].replace("ffmpeg version", "").strip() if ff else None
    return out


def host_info() -> dict:
    return {"host": platform.node(), "os": f"{platform.system()} {platform.release()} ({platform.version()})", "cpu": cpu_name(),
            "cpu_threads": os.cpu_count(), "ram_gib": round((ram_total_bytes() or 0) / GIB, 1), "gpu": gpu_info(), "libs": lib_versions(),
            "collected": datetime.now(timezone.utc).isoformat(timespec="seconds")}


# --- stages ---------------------------------------------------------------------------

def _cuda():
    try:
        import torch
        return torch.cuda if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


class GpuSampler:
    """Samples `nvidia-smi` (GPU utilization %, memory used, power) every `interval` seconds, in a separate thread.
    Stages average the samples that fall inside their window; stages shorter than the interval get no value."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[tuple[float, float, float, float]] = []   # (t perf_counter, util %, mem MiB, W)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = _run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]) is not None

    def start(self):
        if self.available and self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu-sampler")
            self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            out = _run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"])
            if out:
                try:
                    u, m, p = (float(x.strip()) for x in out.splitlines()[0].split(",")[:3])
                    self.samples.append((t, u, m, p))
                except ValueError:
                    pass
            self._stop.wait(max(0.05, self.interval - (time.perf_counter() - t)))

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def window(self, t0: float, t1: float) -> dict | None:
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return None
        return {"gpu_util_mean": round(sum(x[1] for x in s) / len(s), 1), "gpu_util_max": round(max(x[1] for x in s), 1),
                "gpu_mem_used_mib_max": round(max(x[2] for x in s)), "gpu_power_w_mean": round(sum(x[3] for x in s) / len(s), 1), "gpu_samples": len(s)}


class Telemetry:
    def __init__(self, label: str = "", gpu_sampling: bool = True, gpu_interval: float = 1.0):
        self.label = label
        self.stages: list[dict] = []
        self.t_start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.rss_start = rss_bytes()
        self.meta: dict = {}
        self._lock = threading.Lock()
        self._active = 0
        self.gpu = GpuSampler(gpu_interval).start() if gpu_sampling else None

    def close(self):
        if self.gpu is not None:
            self.gpu.stop()

    @contextmanager
    def stage(self, name: str, **info):
        """Times a block. `info` (items, media_s, batch, ...) goes into the record; the block may add fields to `rec`."""
        cu = _cuda()
        with self._lock:
            parallel = self._active > 0
            self._active += 1
        if cu is not None and not parallel:
            cu.synchronize()
            cu.reset_peak_memory_stats()
        rec: dict = {"name": name, "t_offset_s": round(time.perf_counter() - self.t_start, 3)} | info
        t0, c0, r0 = time.perf_counter(), time.process_time(), rss_bytes()
        try:
            yield rec
        finally:
            if cu is not None:
                cu.synchronize()
            wall = time.perf_counter() - t0
            rec["wall_s"] = round(wall, 3)
            rec["cpu_s"] = round(time.process_time() - c0, 3)
            if parallel:
                rec["parallel"] = True
            if cu is not None:
                rec["vram_peak_gib"] = round(cu.max_memory_allocated() / GIB, 3)
                rec["vram_alloc_gib"] = round(cu.memory_allocated() / GIB, 3)
            if self.gpu is not None and (g := self.gpu.window(t0, time.perf_counter())):
                rec |= g
            r1 = rss_bytes()
            if r1 is not None:
                rec["rss_gib"] = round(r1 / GIB, 3)
                if r0 is not None:
                    rec["rss_delta_gib"] = round((r1 - r0) / GIB, 3)
            if rec.get("items"):
                rec["items_per_s"] = round(rec["items"] / wall, 2) if wall > 0 else None
                rec["ms_per_item"] = round(1000 * wall / rec["items"], 1)
            if rec.get("media_s"):
                rec["realtime_factor"] = round(rec["media_s"] / wall, 1) if wall > 0 else None
            with self._lock:
                self._active -= 1
                self.stages.append(rec)

    def timings(self) -> dict[str, float]:
        """Compatibility: {stage: wall seconds}, plus `total`."""
        d = {s["name"]: s["wall_s"] for s in self.stages}
        d["total"] = round(time.perf_counter() - self.t_start, 3)
        return d

    def to_dict(self) -> dict:
        r = rss_bytes()
        gpu = None
        if self.gpu is not None and self.gpu.samples:
            gpu = self.gpu.window(self.t_start, time.perf_counter())
        return {"label": self.label, "meta": self.meta, "stages": self.stages, "gpu_overall": gpu,
                "total_wall_s": round(time.perf_counter() - self.t_start, 3), "total_cpu_s": round(time.process_time() - self.cpu_start, 3),
                "rss_peak_gib": round(max([s.get("rss_gib", 0) for s in self.stages] + [(r or 0) / GIB]), 3),
                "vram_peak_gib": round(max([s.get("vram_peak_gib", 0) for s in self.stages] + [0]), 3)}
