import pytest

from src.report import headline_conditions, resolve_model

CFG = {"models": [{"name": "gpt-4o", "provider": "openai"},
                  {"name": "clip-vit-base-patch32", "provider": "clip"}]}


def test_default_model_is_the_first_in_the_config():
    assert resolve_model(CFG)["name"] == "gpt-4o"


def test_named_model_is_found():
    assert resolve_model(CFG, "clip-vit-base-patch32")["provider"] == "clip"


def test_unknown_model_fails_loudly_rather_than_silently_using_the_default():
    with pytest.raises(SystemExit):
        resolve_model(CFG, "gemini-2.5")


def test_headline_is_clean_plus_every_family_at_full_strength():
    conds = ["clean", "gaussian_noise:50", "gaussian_noise:100", "fisheye:100"]
    assert headline_conditions(conds) == ["clean", "gaussian_noise:100", "fisheye:100"]
