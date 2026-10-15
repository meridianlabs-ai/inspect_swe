"""Unit tests for the Antigravity CLI agent's failure reporting.

When `agy` exits non-zero the only record of why is what this helper returns:
it becomes the RuntimeError message, which is what lands in the Inspect eval
log. Getting it wrong fails SILENTLY -- the run still errors, it just errors
without a cause, and on a large eval set there is nothing left to diagnose
from. That is what these tests exist to catch.

Measured against the real `agy` binary (1.1.20): on failure the CLI writes its
reasoning stream to stdout as ~2KB base64 Gemini thought signatures, one per
turn, and the actual reason ("Error: timeout waiting for response") to stderr.
"""

import json

import pytest
from inspect_swe._antigravity_cli.antigravity_cli import (
    _MAX_ERROR_LEN,
    _clean_antigravity_error,
    _verify_native_result,
)

REASON = "Error: timeout waiting for response"
# Shaped like what agy actually prints: base64 signature then a closing tag on
# its own line. The old filter only dropped lines STARTING with "<think", so
# nothing here was filtered.
SIGNATURE = "Ep8QCpwQAR" + "Zm9vYmFy" * 300


def test_real_reason_survives_a_stdout_full_of_thought_signatures() -> None:
    # Given: many turns' worth of opaque payload on stdout, the reason on stderr
    stdout = "\n".join(f"{SIGNATURE}\n</think>" for _ in range(25))

    # When: the failure output is cleaned for the traceback
    cleaned = _clean_antigravity_error(stdout, f"{REASON}\n")

    # Then: the reason is present and leads, rather than being truncated away.
    assert REASON in cleaned
    assert cleaned.startswith("STDERR:")
    # And: no raw signature survives to crowd it out.
    assert SIGNATURE not in cleaned


def test_opaque_payloads_are_replaced_not_merely_truncated() -> None:
    cleaned = _clean_antigravity_error(SIGNATURE, "")

    assert SIGNATURE not in cleaned
    assert "opaque payload" in cleaned


def test_the_reason_survives_a_stdout_that_overruns_the_budget() -> None:
    # The same failure, with no filter to save it: a CLI that failed late
    # prints far more ordinary output than one error message can carry, and no
    # scrubbing removes prose. Only leading with stderr keeps the reason inside
    # the budget -- putting stdout first is exactly how a failed run once
    # surfaced with no cause in it at all.
    stdout = "\n".join(f"reading file {index}" for index in range(20_000))
    assert len(stdout) > _MAX_ERROR_LEN

    cleaned = _clean_antigravity_error(stdout, f"{REASON}\n")

    assert cleaned.startswith(f"STDERR:\n{REASON}")
    # Bounded, and it says where it stopped rather than leaving a clipped tail
    # the reader cannot tell is missing.
    assert len(cleaned) <= _MAX_ERROR_LEN + len("... (truncated)")
    assert cleaned.endswith("... (truncated)")


def test_ordinary_output_is_preserved_verbatim() -> None:
    cleaned = _clean_antigravity_error("a note on stdout", "a reason on stderr")

    assert "a reason on stderr" in cleaned
    assert "a note on stdout" in cleaned
    # Neither is dropped, and the reason still leads: ordering is the whole
    # reason this helper exists, so it is asserted where nothing is truncated
    # as well as where something is.
    assert cleaned.index("a reason on stderr") < cleaned.index("a note on stdout")


def test_no_output_is_reported_as_such() -> None:
    assert _clean_antigravity_error("", "") == "Unknown error (no output)"


# --- native result verification ---------------------------------------------
#
# A headless run's stdout is the CLI's own JSON result, and process exit status
# alone does not describe it: the CLI can report a failed conversation while
# exiting zero. Two things therefore have to hold before a run is scored or
# returned -- the native status is SUCCESS, and the conversation the result
# names is the one this invocation bound. A post-hoc mismatch is a failure,
# never a fallback: scoring a conversation we did not run is worse than
# erroring, because it produces a number nobody can attribute.
#
# The aggregate `response` in this envelope is deliberately never consulted.
# Canonical output stays the bridge's real `ModelOutput`, messages, IDs and
# usage; this parser only adjudicates identity and status.
#
# The shape below is the factory result envelope of a real 1.1.27 headless run
# -- one JSON object, `conversation_id` in snake case, `status` at the top
# level beside `response`, `duration_seconds`, `num_turns` and `usage`. The
# CLI's STREAM json is a different envelope and is not what this reads.

_RESULT_CID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
_OTHER_CID = "16fd2706-8baf-433b-82eb-8c7fada847da"


def _result(status: str = "SUCCESS", conversation_id: str | None = _RESULT_CID) -> str:
    """The CLI's final JSON result, in the shape a real 1.1.27 run prints."""
    payload: dict[str, object] = {}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    payload["status"] = status
    payload["response"] = "an aggregate the wrapper must not adopt"
    payload["duration_seconds"] = 4.712014882
    payload["num_turns"] = 1
    payload["usage"] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "cache_read_tokens": 0,
        "total_tokens": 0,
    }
    return json.dumps(payload)


def test_a_successful_result_for_the_bound_conversation_verifies() -> None:
    # The only shape that may proceed to scoring.
    _verify_native_result(_result(), _RESULT_CID)


def test_a_result_naming_another_conversation_fails() -> None:
    # The run bound one conversation; a result describing a different one is
    # not evidence about this run at all.
    with pytest.raises(RuntimeError):
        _verify_native_result(_result(conversation_id=_OTHER_CID), _RESULT_CID)


def test_a_result_without_a_conversation_id_fails() -> None:
    # Unverifiable identity is treated as mismatched, not as "probably ours".
    with pytest.raises(RuntimeError):
        _verify_native_result(_result(conversation_id=None), _RESULT_CID)


def test_a_non_success_status_fails_even_though_the_process_exited_zero() -> None:
    # The specific silent path: returncode 0, so nothing in the launch looks
    # wrong, while the CLI's own result says the conversation errored.
    with pytest.raises(RuntimeError):
        _verify_native_result(_result(status="ERROR"), _RESULT_CID)


def test_an_absent_status_fails() -> None:
    payload = json.dumps({"conversation_id": _RESULT_CID})
    with pytest.raises(RuntimeError):
        _verify_native_result(payload, _RESULT_CID)


@pytest.mark.parametrize("stdout", ["", "   ", "not json at all", "{unclosed"])
def test_stdout_that_carries_no_readable_result_fails(stdout: str) -> None:
    # No result is not an implicit success. This is the shape a wedged or
    # truncated run produces, and it must not reach a scorer.
    with pytest.raises(RuntimeError):
        _verify_native_result(stdout, _RESULT_CID)


def test_a_malformed_final_result_is_not_rescued_by_an_earlier_one() -> None:
    # A run that printed a result and then died mid-write leaves a truncated
    # object after it. The FINAL result is the run's result: reaching back past
    # the broken one reports SUCCESS for an attempt whose own result never
    # arrived, which is the same unattributable score a mismatch would produce.
    #
    # There is also no evidence for tolerating a suffix at all -- the real
    # `--output-format json` artifact is exactly one JSON object -- so a second
    # object of any kind means something went wrong that this must not paper
    # over.
    with pytest.raises(RuntimeError):
        _verify_native_result(f"{_result()}\n" + "{unclosed", _RESULT_CID)


def test_the_failure_names_the_conversation_it_expected() -> None:
    # Same reason the rest of this file exists: the raised message is the only
    # record of why, and "identity mismatch" with neither ID in it cannot be
    # diagnosed from an eval log.
    with pytest.raises(RuntimeError) as failure:
        _verify_native_result(_result(conversation_id=_OTHER_CID), _RESULT_CID)

    message = str(failure.value)
    assert _RESULT_CID in message
    assert _OTHER_CID in message


# --- an absent identity is not a matching one -------------------------------
#
# Both sides of the comparison are optional in their own right: a run has no
# bound conversation until the CLI makes its first request, and a result object
# need not carry `conversation_id` at all. Comparing them directly makes those
# two absences agree with each other, so the one run with no identity anywhere
# -- the CLI exiting SUCCESS having never opened a conversation the bridge saw
# -- is the one run that passes unchallenged. It is the worst case to admit,
# because there is no transcript behind it to attribute the score to.


def test_a_result_naming_no_conversation_fails_when_none_was_bound() -> None:
    with pytest.raises(RuntimeError):
        _verify_native_result(_result(conversation_id=None), None)


def test_a_named_conversation_fails_when_none_was_bound() -> None:
    # The other half: the CLI ran something, and this run cannot say it was
    # the thing being scored, because it never saw the conversation open.
    with pytest.raises(RuntimeError):
        _verify_native_result(_result(conversation_id=_RESULT_CID), None)


def test_a_result_naming_no_conversation_fails_when_one_was_bound() -> None:
    with pytest.raises(RuntimeError):
        _verify_native_result(_result(conversation_id=None), _RESULT_CID)


def test_a_non_string_conversation_id_fails() -> None:
    # An id of the wrong type is not a near miss to compare: it means the
    # envelope is not the one this parser adjudicates, whatever it says next.
    payload = json.dumps({"conversation_id": 42, "status": "SUCCESS"})
    with pytest.raises(RuntimeError):
        _verify_native_result(payload, _RESULT_CID)


def test_the_missing_identity_failure_says_which_side_is_missing() -> None:
    # Same reason as the mismatch message: an eval log carrying only "identity
    # mismatch" cannot tell whether the CLI named nothing or this run bound
    # nothing, and those are different faults with different causes.
    with pytest.raises(RuntimeError) as failure:
        _verify_native_result(_result(conversation_id=None), None)

    message = str(failure.value)
    assert "conversation" in message
