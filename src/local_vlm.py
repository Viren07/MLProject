"""Local vision-language models scored the same way as GPT-4o: first-token letter probabilities.

The model is shown the image and the five lettered options, then asked for a single letter. We do
not generate anything. One forward pass gives the distribution over the next token, and the
probabilities of the tokens that spell A to E are read off and renormalised over the offered
letters. That is the same quantity the OpenAI path reads out of top_logprobs, so models from
different providers stay comparable.

Finding the letter tokens is the fiddly part. Tokenizers represent "A" differently depending on
whether it follows a space, and the forms differ per family (A, " A", "ÄŠA" and so on). Rather
than guess, we scan the vocabulary once and keep every token that decodes to a single letter,
then sum their probabilities. `letter_mass` is the total probability on those tokens before
renormalising, and a low value means the model is not answering with a bare letter at all, so the
numbers for that row should not be trusted.

Covers any image-text model transformers can load, which includes Ministral 3, Qwen VL and Gemma.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from .scoring import LETTERS, build_prompt, encode_image

PROMPT_VERSION = "v1"


def letter_token_ids(tokenizer, n_options: int) -> dict[str, list[int]]:
    """Every token id that decodes to a single letter, per letter. Scans the vocab once."""
    wanted = set(LETTERS[:n_options])
    out: dict[str, list[int]] = {L: [] for L in wanted}
    for tok, idx in tokenizer.get_vocab().items():
        try:
            s = tokenizer.convert_tokens_to_string([tok]).strip().upper()
        except Exception:  # noqa: BLE001  - a few tokenizers choke on byte tokens
            continue
        if len(s) == 1 and s in wanted:
            out[s].append(int(idx))
    missing = [L for L, ids in out.items() if not ids]
    if missing:
        raise RuntimeError(f"tokenizer has no single-letter token for {missing}; "
                           "this model cannot be scored by first-token letter probability")
    return out


class LocalVLMScorer:
    """Same .score(image, options) contract as GPTScorer, so src/grid.py needs no changes."""

    def __init__(self, model: str, hf_id: str, cache_dir: str | Path = "results/raw/cache",
                 jpeg_quality: int = 95, load_in_4bit: bool = False, dtype: str = "bfloat16",
                 device: str | None = None, n_options: int = 5, max_image_side: int | None = None,
                 processor_kwargs: dict | None = None):
        import torch
        from transformers import AutoConfig, AutoProcessor

        if load_in_4bit and getattr(AutoConfig.from_pretrained(hf_id), "quantization_config", None):
            raise SystemExit(
                f"{hf_id} already ships quantised weights, so 4-bit loading cannot be applied on "
                "top of them. Point hf_id at the BF16 release of the same model (Mistral publishes "
                "one per Instruct variant, suffixed -BF16) and rerun.")

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name, self.hf_id = model, hf_id
        self.quality, self.four_bit = jpeg_quality, load_in_4bit
        self.max_side = max_image_side
        self.cache_dir = Path(cache_dir) / model

        kwargs: dict = {"dtype": getattr(torch, dtype)}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, dtype), bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = "auto"

        self.processor = AutoProcessor.from_pretrained(hf_id, **(processor_kwargs or {}))
        self.net = self._load(hf_id, kwargs)
        if not load_in_4bit:
            self.net = self.net.to(self.device)
        self.net.eval()

        tok = getattr(self.processor, "tokenizer", self.processor)
        self.letter_ids = letter_token_ids(tok, n_options)

    @staticmethod
    def _load(hf_id: str, kwargs: dict):
        """transformers moved image-text models between auto classes; try the modern one first."""
        import transformers
        errors = []
        for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                return cls.from_pretrained(hf_id, **kwargs)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {type(e).__name__}: {e}")
        raise RuntimeError("could not load " + hf_id + " with any auto class:\n  " + "\n  ".join(errors))

    def _key(self, image_bytes: bytes, options: list[str]) -> str:
        blob = json.dumps({"m": self.hf_id, "pv": PROMPT_VERSION, "q4": self.four_bit,
                           "img": hashlib.sha1(image_bytes).hexdigest(), "opt": options})
        return hashlib.sha1(blob.encode()).hexdigest()

    def _inputs(self, img: Image.Image, options: list[str]):
        """Format as a chat turn ending right where the answer letter would start."""
        prompt = build_prompt(options)
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": prompt}]}]
        try:
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True,
                                                      tokenize=False)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"{self.hf_id} has no usable chat template: {e}") from e
        return self.processor(text=[text], images=[img], return_tensors="pt")

    def score(self, img: Image.Image, options: list[str]) -> dict:
        data = encode_image(img, self.quality)          # same JPEG path as the API models
        key = self._key(data, options)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())

        with Image.open(io.BytesIO(data)) as reopened:
            im = reopened.convert("RGB")
            if self.max_side:
                im.thumbnail((self.max_side, self.max_side))
            inputs = self._inputs(im, options)
        inputs = {k: (v.to(self.net.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with self.torch.no_grad():
            logits = self.net(**inputs).logits[0, -1].float()
        probs = self.torch.softmax(logits, dim=-1)
        raw = [float(probs[self.letter_ids[L]].sum()) for L in LETTERS[:len(options)]]
        mass = sum(raw)
        norm = [r / mass for r in raw] if mass > 0 else [1.0 / len(options)] * len(options)

        out = {"probs": norm, "letter_mass": mass, "served_model": self.hf_id}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        return out
