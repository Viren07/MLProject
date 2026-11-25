"""Letter-token discovery. Loading a real VLM needs a GPU and is not tested here."""
import pytest

from src.local_vlm import letter_token_ids


class FakeTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_string(self, toks):
        return toks[0].replace("_", " ")


def test_finds_bare_and_space_prefixed_forms_of_each_letter():
    tok = FakeTokenizer({"A": 1, "_A": 2, "B": 3, "C": 4, "D": 5, "E": 6, "hello": 9})
    ids = letter_token_ids(tok, 5)
    assert sorted(ids["A"]) == [1, 2]
    assert ids["B"] == [3]


def test_lowercase_counts_as_the_same_letter():
    tok = FakeTokenizer({"A": 1, "a": 2, "B": 3, "C": 4, "D": 5, "E": 6})
    assert sorted(letter_token_ids(tok, 5)["A"]) == [1, 2]


def test_non_letter_tokens_are_ignored():
    tok = FakeTokenizer({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "AB": 6, "!": 7, "": 8})
    ids = letter_token_ids(tok, 5)
    assert all(len(v) == 1 for v in ids.values())


def test_missing_letter_fails_loudly_rather_than_scoring_garbage():
    tok = FakeTokenizer({"A": 1, "B": 2, "C": 3, "D": 4})  # no E
    with pytest.raises(RuntimeError, match="no single-letter token"):
        letter_token_ids(tok, 5)
