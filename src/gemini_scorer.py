"""Gemini scored the same way as GPT-4o: first-token letter probabilities from the API's logprobs.

Only some Gemini versions return logprobs. The 2.x line does; 3.x and later refuse the parameter,
so they cannot be scored this way at all and are excluded from the panel rather than scored by a
different rule. If a model returns a 400 mentioning logprobs, that is what has happened.

Two settings matter beyond the obvious. Thinking is disabled, because a thinking model spends its
first tokens on reasoning and the first token would not be the answer letter. And the output is
capped at one token, because that is the only position we read.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from PIL import Image

from .scoring import OutOfCredits, build_prompt, encode_image, letter_probs

PROMPT_VERSION = "v1"


class LogprobsUnsupported(RuntimeError):
    """This Gemini version refuses logprobs, so it cannot be scored by letter probability."""


def _is_logprobs_refusal(exc: Exception) -> bool:
    t = str(exc).lower()
    return "logprobs" in t and ("not supported" in t or "not enabled" in t or "invalid_argument" in t)


def top_logprobs_from(resp) -> list[tuple[str, float]]:
    """Pull (token, logprob) pairs for the FIRST output token out of a GenerateContentResponse.

    The SDK has moved this field around, so try the documented path and fail loudly with what was
    actually present rather than silently returning an empty distribution.
    """
    cand = (getattr(resp, "candidates", None) or [None])[0]
    result = getattr(cand, "logprobs_result", None) if cand is not None else None
    if result is None:
        raise RuntimeError("no logprobs_result on the response; the model probably ignored "
                           f"response_logprobs. Candidate fields: {dir(cand)[:40] if cand else None}")
    tops = getattr(result, "top_candidates", None)
    if not tops:
        raise RuntimeError(f"logprobs_result has no top_candidates; fields: {dir(result)[:40]}")
    first = tops[0]
    entries = getattr(first, "candidates", None) or []
    out = []
    for e in entries:
        tok = getattr(e, "token", None)
        lp = getattr(e, "log_probability", None)
        if lp is None:
            lp = getattr(e, "logprob", None)
        if tok is not None and lp is not None:
            out.append((tok, float(lp)))
    if not out:
        raise RuntimeError(f"no token/log_probability pairs in the first position; fields: {dir(first)[:40]}")
    return out


class GeminiScorer:
    """Same .score(image, options) contract as GPTScorer."""

    def __init__(self, model: str = "gemini-2.5-flash", cache_dir: str | Path = "results/raw/cache",
                 jpeg_quality: int = 95, top_logprobs: int = 20, client=None, max_retries: int = 6):
        if client is None:
            from google import genai
            key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise SystemExit("set GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env")
            client = genai.Client(api_key=key)
        self.client, self.model = client, model
        self.quality, self.top_k, self.max_retries = jpeg_quality, top_logprobs, max_retries
        self.cache_dir = Path(cache_dir) / model

    def _config(self):
        from google.genai import types
        kw = dict(temperature=0.0, max_output_tokens=1, response_logprobs=True, logprobs=self.top_k)
        try:  # thinking models would spend the first token on reasoning, not on the answer
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:  # noqa: BLE001  - older SDKs have no ThinkingConfig
            pass
        return types.GenerateContentConfig(**kw)

    def _key(self, image_bytes: bytes, options: list[str]) -> str:
        blob = json.dumps({"m": self.model, "pv": PROMPT_VERSION, "k": self.top_k,
                           "img": hashlib.sha1(image_bytes).hexdigest(), "opt": options})
        return hashlib.sha1(blob.encode()).hexdigest()

    def score(self, img: Image.Image, options: list[str]) -> dict:
        data = encode_image(img, self.quality)
        key = self._key(data, options)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())

        resp = self._call(data, options)
        probs, mass = letter_probs(top_logprobs_from(resp), len(options))
        out = {"probs": probs, "letter_mass": mass, "served_model": self.model}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        return out

    def _call(self, image_bytes: bytes, options: list[str]):
        from google.genai import types
        contents = [types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    build_prompt(options)]
        for attempt in range(self.max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=contents, config=self._config())
            except Exception as e:  # noqa: BLE001
                if _is_logprobs_refusal(e):
                    raise LogprobsUnsupported(
                        f"{self.model} refuses logprobs, so it cannot be scored by letter "
                        "probability. Gemini 3.x and later do not support them; use a 2.x model."
                    ) from e
                text = str(e).lower()
                if "quota" in text or "billing" in text or "resource_exhausted" in text:
                    raise OutOfCredits(f"Gemini quota or billing problem: {e}") from e
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 30))
