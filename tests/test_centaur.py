"""Unit tests for the shared centaur runner's user and sandbox handling.

`run_centaur` is the one seam every native CLI wrapper hands its human session
to. Two things a wrapper resolves for its own launch have to reach that session
as well: which user to act as, and which of several sandboxes to act in.
Neither is a preference. A terminal in the wrong container, or opened as the
wrong user, is not the environment the agent would have run in, and the files a
human produces there are not the files the task goes on to score.

The wrapping is asserted here; that it really redirects an unnamed `sandbox()`
lookup is asserted against two live containers in `test_antigravity_cli.py`.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any
from unittest.mock import patch

import anyio
from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.model import ChatMessageUser
from inspect_swe._util import centaur as centaur_module
from inspect_swe._util.centaur import CentaurOptions, run_centaur

_USER = "agent"
_SANDBOX = "target"
_INSTRUCTIONS = "You may also use the CLI via the 'agy' command."
_BASHRC = "alias agy='/var/tmp/agy --print'"


class _RecordedSession:
    """Stands in for the human's terminal, recording what it was handed.

    The trace records the order of the two things that matter: that the session
    was both built and run inside the sandbox selection, and that the selection
    was given back afterwards rather than left set for whatever runs next.
    """

    def __init__(self) -> None:
        self.handed: list[dict[str, Any]] = []
        self.trace: list[str] = []

    def human_cli(self, **kwargs: Any) -> Agent:
        self.handed.append(dict(kwargs))
        self.trace.append("built")

        @agent
        def session() -> Agent:
            async def execute(state: AgentState) -> AgentState:
                self.trace.append("ran")
                return state

            return execute

        return session()

    @contextmanager
    def sandbox_default(self, name: str) -> Iterator[None]:
        self.trace.append(f"selected:{name}")
        try:
            yield
        finally:
            self.trace.append(f"released:{name}")


def _run_centaur(session: _RecordedSession, **kwargs: Any) -> None:
    state = AgentState(messages=[ChatMessageUser(content="Work on the task.")])
    with (
        patch.object(centaur_module, "human_cli", session.human_cli),
        # create=True: the runner has to reach for this by name for a named
        # sandbox to be selectable at all, and the test says which name.
        patch.object(
            centaur_module, "sandbox_default", session.sandbox_default, create=True
        ),
    ):
        anyio.run(
            partial(
                run_centaur,
                CentaurOptions(),
                _INSTRUCTIONS,
                _BASHRC,
                state,
                **kwargs,
            )
        )


def test_the_human_session_runs_as_the_selected_user() -> None:
    # A wrapper told to act as a particular user acts as that user everywhere,
    # or the human is handed a shell that cannot read what the agent wrote.
    session = _RecordedSession()
    _run_centaur(session, user=_USER, sandbox=None)

    assert session.handed, "the human session was never created"
    assert session.handed[0]["user"] == _USER


def test_the_human_session_defaults_to_the_sandbox_user_when_unset() -> None:
    # Withheld, not invented: no selected user means the sandbox's own default,
    # which is what `human_cli` already does with `None`.
    session = _RecordedSession()
    _run_centaur(session, user=None, sandbox=None)

    assert session.handed[0]["user"] is None


def test_the_whole_human_session_runs_in_the_named_sandbox() -> None:
    # Both halves are inside the selection: `human_cli` installs the task tools
    # when it is built, and offers the login when it runs, and each resolves
    # the default sandbox for itself.
    session = _RecordedSession()
    _run_centaur(session, user=None, sandbox=_SANDBOX)

    assert session.trace == [
        f"selected:{_SANDBOX}",
        "built",
        "ran",
        f"released:{_SANDBOX}",
    ]


def test_an_unnamed_sandbox_leaves_the_ambient_default_alone() -> None:
    # The single-container case, which is nearly every task: there is nothing
    # to select, and a selection made anyway would name a sandbox the caller
    # never asked for.
    session = _RecordedSession()
    _run_centaur(session, user=None, sandbox=None)

    assert session.trace == ["built", "ran"]


def test_the_session_still_gets_its_options_and_shell_content() -> None:
    # The rest of what the runner has always forwarded, asserted so that adding
    # the two above cannot quietly drop any of it.
    session = _RecordedSession()
    _run_centaur(session, user=_USER, sandbox=_SANDBOX)

    handed = session.handed[0]
    options = CentaurOptions()
    assert handed["answer"] == options.answer
    assert handed["intermediate_scoring"] == options.intermediate_scoring
    assert handed["record_session"] == options.record_session
    assert handed["instructions"] == _INSTRUCTIONS
    assert handed["bashrc"] == _BASHRC
