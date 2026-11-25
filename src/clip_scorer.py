"""CLIP baseline: probabilities from image-text similarity, with no letters to read.

CLIP measures how well an image matches a text prompt. Prompting it with the five offered
room names and taking a softmax over those five similarities gives a probability per option,
on the same footing as the letter probabilities read out of a VLM, so the conformal code
runs unchanged. There is no first token and nothing to renormalize, so `letter_mass` is
always 1.0 and the low-mass diagnostic never fires.

The image goes through the same JPEG re-encode as the API models. Without it CLIP would be
scoring slightly cleaner pictures than the model it is being compared against.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .scoring import encode_image, pretty

DEFAULT_HF_ID = "openai/clip-vit-base-patch32"
PROMPT_TEMPLATE = "a photo of a {}"


def softmax(x) -> list[float]:
    a = np.asarray(x, dtype=float)
    e = np.exp(a - a.max())
    return (e / e.sum()).tolist()


def build_prompts(options: list[str], template: str = PROMPT_TEMPLATE) -> list[str]:
    return [template.format(pretty(o)) for o in options]


def projected(out, expected_dim: int | None = None):
    """Unwrap the projected embedding from get_image_features / get_text_features.

    transformers 5 returns a BaseModelOutputWithPooling holding the projected features in
    `pooler_output`; transformers 4 returned the tensor itself. The width check is the point
    of this function: the vision tower's unprojected hidden states are a different width and
    would silently corrupt every score rather than raising.
    """
    feat = out if hasattr(out, "norm") else getattr(out, "pooler_output", None)
    if feat is None or not hasattr(feat, "norm"):
        raise RuntimeError(f"CLIP returned {type(out).__name__}, which holds no usable embedding.")
    if expected_dim is not None and feat.shape[-1] != expected_dim:
        raise RuntimeError(
            f"CLIP returned features of width {feat.shape[-1]}, expected {expected_dim}. "
            "The transformers feature API has changed again; these are probably unprojected "
            "hidden states, which would corrupt every score silently.")
    return feat


class CLIPScorer:
    """Same .score(image, options) contract as GPTScorer, so src/grid.py needs no changes."""

    def __init__(self, model: str = "clip-vit-base-patch32", hf_id: str = DEFAULT_HF_ID,
                 cache_dir: str | Path = "results/raw/cache", jpeg_quality: int = 95,
                 prompt_template: str = PROMPT_TEMPLATE, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name, self.hf_id = model, hf_id
        self.quality, self.template = jpeg_quality, prompt_template
        self.cache_dir = Path(cache_dir) / model
        self.net = CLIPModel.from_pretrained(hf_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(hf_id)
        self.dim = int(self.net.config.projection_dim)
        self._text_cache: dict[str, "torch.Tensor"] = {}

    def _key(self, image_bytes: bytes, options: list[str]) -> str:
        blob = json.dumps({"m": self.hf_id, "t": self.template,
                           "img": hashlib.sha1(image_bytes).hexdigest(), "opt": options})
        return hashlib.sha1(blob.encode()).hexdigest()

    def _text_features(self, options: list[str]):
        """Encode each option name once and reuse it; the 67 class names recur constantly."""
        missing = [o for o in options if o not in self._text_cache]
        if missing:
            batch = self.processor(text=build_prompts(missing, self.template),
                                   return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                feats = projected(self.net.get_text_features(**batch), self.dim)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            for name, row in zip(missing, feats):
                self._text_cache[name] = row
        return self.torch.stack([self._text_cache[o] for o in options])

    def score(self, img: Image.Image, options: list[str]) -> dict:
        data = encode_image(img, self.quality)
        key = self._key(data, options)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())

        with Image.open(io.BytesIO(data)) as reopened:
            inputs = self.processor(images=reopened.convert("RGB"),
                                    return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            feat = projected(self.net.get_image_features(**inputs), self.dim)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logits = (self.net.logit_scale.exp() * feat @ self._text_features(options).T)
        out = {"probs": softmax(logits[0].float().cpu().numpy()), "letter_mass": 1.0,
               "served_model": self.hf_id}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        return out
