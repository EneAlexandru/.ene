"""The embedding model of profile `ene-1` (SPEC §6.1, §7): EmbeddingGemma 2 via sentence-transformers: text, image, audio,
video, interleaved. The one module that names a model: everything else takes an `Embedder` and reads its `name`, `model_id`,
`space_id`, `dim`, `space()` and `compute()`; another model is another module with this interface and another space id (§2.7).

Rules from the documentation: bf16 or fp32 (never fp16); prompts only on text; media without a prefix;
mono 16 kHz audio with an explicit `max_length` (by default the model only sees the first 30 s);
image `max_soft_tokens` in {70,140,280,560,1120}; video `max_frames` + `overflow_strategy`.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

NAME = "EmbeddingGemma 2"
MODEL_ID = "google/embeddinggemma-2"
SPACE_ID = "eg2-768"                  # the model at its native width (§2.7); a fine-tune or another model is another id
SR = 16000


class Embedder:
    def __init__(self, model_id: str = MODEL_ID, space_id: str = SPACE_ID, device: str | None = None, text_only: bool = False):
        from sentence_transformers import SentenceTransformer

        self.name, self.model_id, self.space_id = NAME, model_id, space_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        cfg = {"vision_config": None, "audio_config": None} if text_only else {}
        t0 = time.perf_counter()
        self.model = SentenceTransformer(model_id, device=self.device, model_kwargs={"torch_dtype": self.dtype}, config_kwargs=cfg)
        self.load_s = time.perf_counter() - t0
        self.dim = self.model.get_embedding_dimension()
        self.revision = self._revision(model_id)

    @staticmethod
    def _revision(model_id: str) -> str | None:
        try:
            from huggingface_hub import scan_cache_dir
            for repo in scan_cache_dir().repos:
                if repo.repo_id == model_id:
                    main = next((r for r in repo.revisions if "main" in r.refs), None)      # what `main` resolves to offline
                    return (main or sorted(repo.revisions, key=lambda r: r.last_modified)[-1]).commit_hash
        except Exception:  # noqa: BLE001
            return None
        return None

    def space(self) -> dict:
        """The `spaces` entry of the description (SPEC §3.7): model, revision, width, pooling, normalization, library versions."""
        import sentence_transformers, transformers
        return {
            "model": self.model_id, "revision": self.revision, "dim": self.dim, "mrl": [768, 512, 256, 128],
            "normalized": True, "pooling": "mean", "modalities": ["text", "image", "video", "audio"],
            "libs": {"sentence-transformers": sentence_transformers.__version__, "transformers": transformers.__version__, "torch": torch.__version__},
        }

    def _enc(self, inputs, batch_size: int, **kw) -> np.ndarray:
        out = self.model.encode(inputs, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, **kw)
        return np.ascontiguousarray(out, dtype=np.float32).reshape(-1, self.dim)

    # --- text ---
    def text(self, texts: list[str], prompt_name: str | None = "Document", prompt: str | None = None, batch_size: int = 32) -> np.ndarray:
        kw = {"prompt": prompt} if prompt is not None else {"prompt_name": prompt_name}
        return self._enc(list(texts), batch_size, **kw)

    def documents(self, texts: list[str], title: str | None) -> np.ndarray:
        """Prompt `title: {title} | text: ` (per the documentation, the real title improves quality)."""
        return self.text(texts, prompt=f"title: {title or 'none'} | text: ")

    def queries(self, texts: list[str]) -> np.ndarray:
        return self.text(texts, prompt_name="SearchQuery")

    # --- image ---
    def images(self, imgs: list[Image.Image], max_soft_tokens: int = 280, batch_size: int = 16) -> np.ndarray:
        return self._enc([{"image": im} for im in imgs], batch_size, processing_kwargs={"image": {"max_soft_tokens": max_soft_tokens}})

    # --- audio ---
    def audio(self, clips: list[np.ndarray], sr: int = SR, batch_size: int = 8) -> np.ndarray:
        """Mono float32 clips; `max_length` = the longest clip in the batch, so nothing is silently truncated."""
        out = []
        for i in range(0, len(clips), batch_size):
            batch = clips[i:i + batch_size]
            max_len = max(len(c) for c in batch)
            inputs = [{"audio": {"array": np.ascontiguousarray(c, dtype=np.float32), "sampling_rate": sr}} for c in batch]
            out.append(self._enc(inputs, batch_size, processing_kwargs={"audio": {"max_length": int(max_len), "sampling_rate": sr}}))
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    # --- video ---
    def video(self, clips: list[list[Image.Image]], max_soft_tokens: int = 70, max_frames: int = 32, batch_size: int = 2) -> np.ndarray:
        """Each clip = a list of frames (already at ~1 fps). Without metadata, the processor keeps up to `max_frames` frames (uniform)."""
        inputs = [{"video": frames} for frames in clips]
        return self._enc(inputs, batch_size, processing_kwargs={"video": {"max_soft_tokens": max_soft_tokens, "max_frames": max_frames, "fps": None, "overflow_strategy": "uniform"}})

    # --- interleaved ---
    def interleaved(self, text: str, images: list[Image.Image] | None = None, audio: list[np.ndarray] | None = None,
                    video: list[Image.Image] | None = None, image_tokens: int = 280, video_tokens: int = 70, max_frames: int = 32, sr: int = SR) -> np.ndarray:
        """A single vector from text + `<|image|>` x n + `<|audio|>` x m + `<|video|>`; the placeholders must appear in the text."""
        d: dict = {"text": text}
        pk: dict = {"image": {"max_soft_tokens": image_tokens}, "video": {"max_soft_tokens": video_tokens, "max_frames": max_frames, "fps": None, "overflow_strategy": "uniform"}}
        if images:
            d["image"] = list(images)
        if audio:
            d["audio"] = [{"array": np.ascontiguousarray(a, dtype=np.float32), "sampling_rate": sr} for a in audio]
            pk["audio"] = {"max_length": int(max(len(a) for a in audio)), "sampling_rate": sr}
        if video:
            d["video"] = list(video)
        return self._enc(d, 1, processing_kwargs=pk)

    def compute(self) -> dict:
        """`generator.compute` of the description (§3.7): informative, never part of `prep` (§8.4)."""
        return {"device": self.device, "dtype": str(self.dtype).replace("torch.", "")}

    def vram(self) -> float:
        return torch.cuda.max_memory_allocated() / 2**30 if self.device == "cuda" else 0.0
