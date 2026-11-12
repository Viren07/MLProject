"""Prompt building, GPT-4o letter probabilities, and a disk cache.

For each image we ask a 5-option question and read the model's probability of each
answer letter from the first output token's top log-probabilities (KnowNo style).
Probabilities are renormalized over the offered letters. `letter_mass` (raw mass
before renormalizing) is kept as a diagnostic: if it is low, the model is not
answering with a bare letter and the numbers are not trustworthy.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import time
from pathlib import Path

from PIL import Image

LETTERS = "ABCDEFGH"
PROMPT_VERSION = "v1"


def pretty(cls: str) -> str:
    return cls.replace("_", " ")


def build_prompt(options: list[str]) -> str:
    lines = "\n".join(f"{LETTERS[i]}. {pretty(o)}" for i, o in enumerate(options))
    return ("This is a photo taken inside a building. Which type of room or place is it?\n"
            f"{lines}\n"
            "Answer with the single letter of the best option and nothing else.")


def encode_image(img: Image.Image, quality: int = 95) -> bytes:
    """One encoding path for clean AND corrupted images, so the codec is never a confound."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def build_messages(image_bytes: bytes, options: list[str], detail: str = "high") -> list[dict]:
    url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
    return [{"role": "user", "content": [
        {"type": "text", "text": build_prompt(options)},
        {"type": "image_url", "image_url": {"url": url, "detail": detail}},
    ]}]


def letter_probs(top_logprobs: list[tuple[str, float]], n_options: int) -> tuple[list[float], float]:
    """top_logprobs: (token, logprob) pairs for the first output token.
    Returns (probabilities over the n letters, summing to 1; raw letter mass)."""
    raw = [0.0] * n_options
    for tok, lp in top_logprobs:
        t = tok.strip().upper().rstrip(".)")
        if len(t) == 1 and t in LETTERS[:n_options]:
            raw[LETTERS.index(t)] += math.exp(lp)
    mass = sum(raw)
    if mass <= 0:
        return [1.0 / n_options] * n_options, 0.0
    return [r / mass for r in raw], mass


class OutOfCredits(RuntimeError):
    """The API account has no balance left. Retrying is pointless; add credits and rerun (cached calls are free)."""


def _is_out_of_credits(exc: Exception) -> bool:
    text = str(exc).lower()
    return "insufficient_quota" in text or "no credits remaining" in text or "exceeded your current quota" in text


class GPTScorer:
    def __init__(self, model: str = "gpt-4o", cache_dir: str | Path = "results/raw/cache",
                 detail: str = "high", jpeg_quality: int = 95, client=None, max_retries: int = 6):
        if client is None:
            from openai import OpenAI  # reads OPENAI_API_KEY
            client = OpenAI()
        self.client, self.model, self.detail = client, model, detail
        self.quality, self.max_retries = jpeg_quality, max_retries
        self.cache_dir = Path(cache_dir) / model

    def _key(self, image_bytes: bytes, options: list[str]) -> str:
        blob = json.dumps({"m": self.model, "pv": PROMPT_VERSION, "d": self.detail,
                           "img": hashlib.sha1(image_bytes).hexdigest(), "opt": options})
        return hashlib.sha1(blob.encode()).hexdigest()

    def score(self, img: Image.Image, options: list[str]) -> dict:
        data = encode_image(img, self.quality)
        key = self._key(data, options)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())

        resp = self._call(build_messages(data, options, self.detail))
        ch = resp.choices[0]
        tops = [(t.token, t.logprob) for t in ch.logprobs.content[0].top_logprobs]
        probs, mass = letter_probs(tops, len(options))
        out = {"probs": probs, "letter_mass": mass, "first_token": ch.logprobs.content[0].token,
               "served_model": getattr(resp, "model", None),
               "fingerprint": getattr(resp, "system_fingerprint", None)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        return out

    def _call(self, messages):
        for attempt in range(self.max_retries):
            try:
                return self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=1, temperature=0,
                    logprobs=True, top_logprobs=20)
            except Exception as e:  # noqa: BLE001
                if _is_out_of_credits(e):
                    raise OutOfCredits("OpenAI account is out of credits. Add credits at "
                                       "https://platform.openai.com/settings/organization/billing/ "
                                       "and rerun; calls already made are cached and cost nothing.") from e
                name = type(e).__name__
                fatal = name in {"AuthenticationError", "PermissionDeniedError", "BadRequestError",
                                 "NotFoundError"}
                if fatal or attempt == self.max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 30))
