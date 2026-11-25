"""The CLIP baseline's pure logic. Loading the network needs torch and is not tested here."""
from types import SimpleNamespace

import pytest

from src.clip_scorer import build_prompts, projected, softmax


def test_prompts_use_readable_class_names():
    assert build_prompts(["inside_subway"]) == ["a photo of a inside subway"]


def test_prompt_template_is_configurable():
    assert build_prompts(["bakery"], "an indoor photo of a {}") == ["an indoor photo of a bakery"]


def test_softmax_is_a_distribution_over_the_offered_options():
    p = softmax([3.0, 1.0, 1.0])
    assert pytest.approx(sum(p)) == 1.0
    assert len(p) == 3
    assert p[0] > p[1] == p[2]


def test_softmax_survives_clip_scale_logits():
    """CLIP multiplies cosine similarities by a learned scale near 100, so logits get large."""
    p = softmax([800.0, 799.0, 10.0])
    assert pytest.approx(sum(p)) == 1.0
    assert all(x == x for x in p)  # no NaN from overflow


class _FakeTensor:
    """Stands in for a torch tensor: the unwrap only checks for .norm and .shape."""

    def __init__(self, width: int = 512):
        self.shape = (1, width)

    def norm(self, **kwargs):
        return self


def test_transformers_4_style_tensor_passes_through():
    t = _FakeTensor()
    assert projected(t, 512) is t


def test_transformers_5_style_output_is_unwrapped():
    t = _FakeTensor()
    assert projected(SimpleNamespace(pooler_output=t), 512) is t


def test_wrong_width_raises_instead_of_corrupting_every_score():
    with pytest.raises(RuntimeError, match="width 768"):
        projected(SimpleNamespace(pooler_output=_FakeTensor(768)), 512)


def test_output_with_no_embedding_raises():
    with pytest.raises(RuntimeError, match="no usable embedding"):
        projected(SimpleNamespace(last_hidden_state=_FakeTensor()), 512)
