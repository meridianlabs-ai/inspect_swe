"""codex_cli launches codex with a prompt longer than one argv entry may be.

The prompt was the last argv entry of `codex exec`, and Linux caps a single argv
entry at 128 KiB (MAX_ARG_STRLEN), so a longer prompt failed to launch with
`OSError(7, 'Argument list too long')`. codex now reads the prompt from stdin.
mockllm answers, so the test needs a sandbox but no model API.
"""

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import resolve_sample_attachments
from inspect_ai.model import ChatMessageUser
from inspect_swe import codex_cli

from tests.conftest import assert_eval_completed, skip_if_no_docker

# over the 128 KiB argv entry limit, with characters a shell or an argv parser
# could alter: quotes, `$`, backticks, non-ASCII, and a leading dash
PROMPT = '-x "quoted" $HOME `tick` ünïcødé 😀\n\nReply with: done.\n' + "x" * 200_000


@pytest.mark.slow
@skip_if_no_docker
def test_codex_cli_delivers_a_prompt_longer_than_an_argv_entry() -> None:
    task = Task(dataset=[Sample(input=PROMPT)], solver=codex_cli(), sandbox="docker")

    log = eval(task, model="mockllm/model", time_limit=300)[0]

    sample = resolve_sample_attachments(assert_eval_completed(log)[0])
    sent = [
        message.text
        for event in sample.events
        if event.event == "model"
        for message in event.input
        if isinstance(message, ChatMessageUser)
    ]
    assert PROMPT in sent
