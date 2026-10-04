"""opencode receives its prompt on stdin, not as a positional argument.

``opencode run`` quote-wraps a positional message that contains spaces and
backslash-escapes the double quotes inside it (``packages/opencode/src/cli/cmd/
run.ts``), so the prompt the model saw -- and the user message crossing the
agent bridge -- differed from the task input. The bridge anchors main-thread
tracking on the task input, and for prompts containing ``"`` the mismatch let
opencode's session-title generation call be surfaced as the sample's final
answer (GAIA level 1, opencode 1.18.29 + gpt-5.5: 4 of 10 samples). Piped stdin
is used verbatim, so the prompt is delivered via ``exec_remote(input=...)``.

The agent's sandbox plumbing is faked at the module level so the test drives
``execute()`` itself and inspects the exact ``exec_remote`` call it makes.
"""

import importlib
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import anyio
import pytest
from inspect_ai.agent import AgentState
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

# the package re-exports the agent function under the module's name
opencode_module = importlib.import_module("inspect_swe._opencode.opencode")

# embedded double quotes, blank lines, and no trailing newline
PROMPT = 'Write the opposite of the word "left".\n\nAnswer with one word'


class FakeResult:
    success = True
    returncode = 0
    stdout = ""
    stderr = ""


class FakeSandbox:
    def __init__(self) -> None:
        self.exec_remote_calls: list[dict[str, Any]] = []
        self.written: dict[str, str] = {}

    async def exec(self, cmd: list[str], **kwargs: Any) -> FakeResult:
        result = FakeResult()
        if cmd[:2] == ["sh", "-c"] and "HOME" in cmd[2]:
            result.stdout = "/root\n"
        return result

    async def write_file(self, path: str, contents: str) -> None:
        self.written[path] = contents

    async def exec_remote(
        self, cmd: list[str], options: Any, stream: bool
    ) -> FakeResult:
        self.exec_remote_calls.append({"cmd": cmd, "options": options})
        return FakeResult()


class FakeBridge:
    port = 3001
    mcp_server_configs: list[Any] = []

    def __init__(self, state: AgentState) -> None:
        self.state = state


class FakeStore:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value


def run_opencode(
    monkeypatch: pytest.MonkeyPatch, messages: list[Any], **kwargs: Any
) -> FakeSandbox:
    sbox = FakeSandbox()
    store = FakeStore()

    @asynccontextmanager
    async def fake_bridge(state: AgentState, **_: Any) -> AsyncIterator[FakeBridge]:
        yield FakeBridge(state)

    async def fake_cwd(*_: Any) -> str:
        return "/root"

    async def fake_setup(*_: Any) -> tuple[str, list[str]]:
        return "/opt/opencode/opencode", []

    monkeypatch.setattr(opencode_module, "sandbox_env", lambda *_: sbox)
    monkeypatch.setattr(opencode_module, "sandbox_agent_bridge", fake_bridge)
    monkeypatch.setattr(opencode_module, "resolve_agent_cwd", fake_cwd)
    monkeypatch.setattr(opencode_module, "ensure_opencode_setup", fake_setup)
    monkeypatch.setattr(opencode_module, "store", lambda: store)

    agent = opencode_module.opencode(**kwargs)
    anyio.run(agent, AgentState(messages=messages))
    return sbox


def test_prompt_is_delivered_on_stdin_not_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    sbox = run_opencode(monkeypatch, [ChatMessageUser(content=PROMPT)])

    (call,) = sbox.exec_remote_calls
    options = call["options"]
    assert isinstance(options, ExecRemoteAwaitableOptions)
    # verbatim: embedded quotes, blank lines, no trailing newline
    assert options.input == PROMPT
    # opencode's own argv, with no message appended (and no shell wrapper)
    assert call["cmd"][0] == "/opt/opencode/opencode"
    assert call["cmd"][1] == "run"
    assert PROMPT not in call["cmd"]
    assert not any(PROMPT in arg for arg in call["cmd"])
    # nothing is staged on disk apart from opencode's config
    assert list(sbox.written) == ["/root/.config/opencode/opencode.json"]


def test_system_prompt_is_prepended_within_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sbox = run_opencode(
        monkeypatch,
        [ChatMessageSystem(content="Be terse."), ChatMessageUser(content=PROMPT)],
        system_prompt="Answer in English.",
    )

    (call,) = sbox.exec_remote_calls
    assert call["options"].input == f"Be terse.\n\nAnswer in English.\n\n{PROMPT}"
    assert "--continue" not in call["cmd"]


def _title_args(cmd: list[str]) -> list[str]:
    return [arg for arg in cmd if arg.startswith("--title")]


def test_default_session_title_skips_title_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # By default a fixed title is passed as a single `--title=<value>`
    # argument, which makes opencode's session title non-default and skips
    # its automatic title-generation model call.
    sbox = run_opencode(monkeypatch, [ChatMessageUser(content=PROMPT)])

    (call,) = sbox.exec_remote_calls
    assert _title_args(call["cmd"]) == ["--title=Inspect eval"]


def test_session_title_none_restores_title_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `None` opts back in to opencode's normal title generation: no `--title`
    # argument of any form is emitted.
    sbox = run_opencode(
        monkeypatch, [ChatMessageUser(content=PROMPT)], session_title=None
    )

    (call,) = sbox.exec_remote_calls
    assert _title_args(call["cmd"]) == []


def test_session_title_is_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    sbox = run_opencode(
        monkeypatch, [ChatMessageUser(content=PROMPT)], session_title="my run"
    )

    (call,) = sbox.exec_remote_calls
    assert _title_args(call["cmd"]) == ["--title=my run"]


def test_session_title_is_passed_on_continuation_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `--title` coexists with `--continue`; opencode ignores it when resuming.
    sbox = run_opencode(
        monkeypatch,
        [
            ChatMessageUser(content="first"),
            ChatMessageAssistant(content="ok"),
            ChatMessageUser(content=PROMPT),
        ],
        session_title="my run",
    )

    (call,) = sbox.exec_remote_calls
    cmd = call["cmd"]
    assert "--continue" in cmd
    assert _title_args(cmd) == ["--title=my run"]


def test_dash_prefixed_session_title_is_not_parsed_as_a_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `["--title", "-draft"]` as two argv entries would make opencode's yargs
    # parser treat "-draft" as an unknown option rather than a literal title.
    # A single `--title=-draft` argument is unambiguous.
    sbox = run_opencode(
        monkeypatch, [ChatMessageUser(content=PROMPT)], session_title="-draft"
    )

    (call,) = sbox.exec_remote_calls
    cmd = call["cmd"]
    # the value appears exactly once, fused to the flag -- never as its own
    # argv entry that a parser could misread as an option
    assert [arg for arg in cmd if "draft" in arg] == ["--title=-draft"]


def test_continue_prefixed_session_title_does_not_trigger_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `["--title", "--continue"]` as two argv entries would let opencode's
    # parser read "--continue" as its own flag (leaving the title empty) and
    # make a fresh run resume an unrelated prior session. Encoding the title
    # as one `--title=<value>` argument keeps it a literal value.
    sbox = run_opencode(
        monkeypatch, [ChatMessageUser(content=PROMPT)], session_title="--continue"
    )

    (call,) = sbox.exec_remote_calls
    cmd = call["cmd"]
    assert _title_args(cmd) == ["--title=--continue"]
    assert cmd.count("--continue") == 0


def test_session_title_applies_in_centaur_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Centaur mode aliases the same opencode command for the human operator;
    # the title travels with it so the option means the same thing everywhere.
    captured: dict[str, str] = {}

    async def fake_run_centaur(
        options: Any, instructions: str, bashrc: str, state: AgentState
    ) -> None:
        captured["bashrc"] = bashrc

    monkeypatch.setattr(opencode_module, "run_centaur", fake_run_centaur)
    sbox = run_opencode(monkeypatch, [ChatMessageUser(content=PROMPT)], centaur=True)

    assert sbox.exec_remote_calls == []
    assert "alias opencode=" in captured["bashrc"]
    assert "--title=Inspect eval" in captured["bashrc"]
    assert "--dangerously-skip-permissions" not in captured["bashrc"]


def test_continuation_turn_uses_stdin_too(monkeypatch: pytest.MonkeyPatch) -> None:
    sbox = run_opencode(
        monkeypatch,
        [
            ChatMessageUser(content="first"),
            ChatMessageAssistant(content="ok"),
            ChatMessageUser(content=PROMPT),
        ],
    )

    (call,) = sbox.exec_remote_calls
    assert "--continue" in call["cmd"]
    assert call["options"].input == PROMPT
    assert PROMPT not in call["cmd"]
