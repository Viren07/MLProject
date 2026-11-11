import math
from types import SimpleNamespace as NS

from PIL import Image

from src.scoring import GPTScorer, build_prompt, letter_probs


def test_prompt_lists_letters_and_cleans_names():
    p = build_prompt(["airport_inside", "bakery", "bar", "casino", "closet"])
    assert "A. airport inside" in p and "E. closet" in p and "_" not in p


def test_letter_probs_renormalizes_and_merges_variants():
    tops = [("A", math.log(0.6)), (" A", math.log(0.2)), ("B", math.log(0.1)), ("The", math.log(0.05))]
    probs, mass = letter_probs(tops, 5)
    assert abs(sum(probs) - 1) < 1e-9 and abs(mass - 0.9) < 1e-9
    assert abs(probs[0] - 0.8 / 0.9) < 1e-9 and probs[4] == 0.0


def test_letter_probs_ignores_letters_outside_options_and_handles_no_mass():
    probs, mass = letter_probs([("F", math.log(0.9))], 5)
    assert mass == 0.0 and probs == [0.2] * 5


class FakeClient:
    def __init__(self):
        self.calls = 0
        outer = self

        class C:
            def create(self, **kw):
                outer.calls += 1
                tops = [NS(token="C", logprob=math.log(0.7)), NS(token="A", logprob=math.log(0.3))]
                content = [NS(token="C", top_logprobs=tops)]
                return NS(choices=[NS(logprobs=NS(content=content))], model="fake", system_fingerprint="fp")
        self.chat = NS(completions=C())


def test_scorer_caches_to_disk(tmp_path):
    client = FakeClient()
    s = GPTScorer(cache_dir=tmp_path, client=client)
    img = Image.new("RGB", (64, 64), "gray")
    opts = ["a", "b", "c", "d", "e"]
    r1 = s.score(img, opts)
    r2 = s.score(img, opts)
    assert client.calls == 1 and r1 == r2
    assert r1["probs"][2] > 0.69 and abs(sum(r1["probs"]) - 1) < 1e-9
    s.score(img, opts[::-1])  # different option order must not reuse the cache
    assert client.calls == 2
