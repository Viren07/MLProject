"""Response parsing and error classification. Live calls are not tested here."""
from types import SimpleNamespace as NS

import pytest

from src.gemini_scorer import _is_logprobs_refusal, top_logprobs_from


def response_with(pairs):
    entries = [NS(token=t, log_probability=lp) for t, lp in pairs]
    return NS(candidates=[NS(logprobs_result=NS(top_candidates=[NS(candidates=entries)]))])


def test_reads_first_position_token_logprob_pairs():
    assert top_logprobs_from(response_with([("A", -0.1), (" B", -2.3)])) == [("A", -0.1), (" B", -2.3)]


def test_missing_logprobs_result_says_so_instead_of_returning_nothing():
    with pytest.raises(RuntimeError, match="no logprobs_result"):
        top_logprobs_from(NS(candidates=[NS(text="A")]))


def test_empty_top_candidates_is_an_error():
    with pytest.raises(RuntimeError, match="no top_candidates"):
        top_logprobs_from(NS(candidates=[NS(logprobs_result=NS(top_candidates=[]))]))


def test_a_logprobs_refusal_is_recognised():
    assert _is_logprobs_refusal(Exception("400 Logprobs is not supported for this model"))
    assert _is_logprobs_refusal(Exception("Logprobs is not enabled for this model"))


def test_an_ordinary_error_is_not_mistaken_for_a_refusal():
    assert not _is_logprobs_refusal(Exception("503 Service Unavailable"))
