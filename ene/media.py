"""Media decoding with ffmpeg: mono 16 kHz float32 audio, frames at a fixed fps, frames at scene cuts (SPEC §6.1)."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
from PIL import Image

SR = 16000


def audio_f32(path: str, stream: int = 0, sr: int = SR) -> np.ndarray:
    """The whole audio track as mono float32 at `sr` Hz (decoded once; windows are sliced from memory)."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-map", f"0:a:{stream}", "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<f4")


def windows(n_samples: int, win_s: float, stride_s: float, sr: int = SR) -> list[tuple[int, int]]:
    """Windows [start, end) in samples; the last window is shortened, not dropped."""
    win, step = int(win_s * sr), int(stride_s * sr)
    out, s = [], 0
    while s < n_samples:
        e = min(s + win, n_samples)
        out.append((s, e))
        if e == n_samples:
            break
        s += step
    return out


def _scaled_size(width: int, height: int, target_w: int) -> tuple[int, int]:
    h = int(round(height * target_w / width / 2)) * 2
    return target_w, h


@dataclass
class Frame:
    t: float                 # seconds from the start of the file (exact from ffmpeg; the writer turns it into a frame index)
    image: Image.Image

    t_ms = property(lambda self: int(round(self.t * 1000)))


def frames_at_fps(path: str, width: int, height: int, fps: float = 1.0, target_w: int = 448) -> list[Frame]:
    """Frames at a fixed `fps` (the first at t=0), resized to width `target_w`, as PIL RGB."""
    w, h = _scaled_size(width, height, target_w)
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps},scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(out) // (w * h * 3)
    arr = np.frombuffer(out[: n * w * h * 3], dtype=np.uint8).reshape(n, h, w, 3)
    return [Frame(i / fps, Image.fromarray(arr[i])) for i in range(n)]


def scene_frames(path: str, width: int, height: int, threshold: float = 0.3, target_w: int = 448, min_gap_s: float = 1.0) -> list[Frame]:
    """The frame right after each scene cut (ffmpeg `select=gt(scene,T)`), with the exact time from `showinfo`."""
    w, h = _scaled_size(width, height, target_w)
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False, mode="w+", encoding="utf-8") as log:
        cmd = ["ffmpeg", "-v", "info", "-nostats", "-i", path, "-vf", f"select='gt(scene,{threshold})',showinfo,scale={w}:{h}",
               "-fps_mode", "vfr", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, check=True).stdout
        log.seek(0)
        times = [float(m) for m in re.findall(r"pts_time:\s*([0-9.]+)", log.read())]
    n = len(out) // (w * h * 3)
    arr = np.frombuffer(out[: n * w * h * 3], dtype=np.uint8).reshape(n, h, w, 3)
    frames, last = [], -1e9
    for i in range(min(n, len(times))):
        if times[i] - last >= min_gap_s:
            frames.append(Frame(times[i], Image.fromarray(arr[i])))
            last = times[i]
    return frames


def pick_uniform(items: list, k: int) -> list:
    """k uniformly spread elements (like `overflow_strategy: uniform`)."""
    if len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]
