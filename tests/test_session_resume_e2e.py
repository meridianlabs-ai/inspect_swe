"""End-to-end session-resume tests (real CLI, real adapter, real sandbox).

Two tiers, because they fail for different reasons and cost different amounts:

1. **Load tier** (docker only, no model call). Starts the agent, waits for the
   ACP session to open, and then inspects the sandbox: the synthesized session
   file has to be exactly where the CLI looks for it, and the CLI process has to
   have been launched resuming it. This is where the things that actually break
   live — the on-disk path (config dir, cwd slug, symlink resolution), the
   serialized content, and whether ``session/load`` accepted it at all (if it
   didn't, ``ready`` never fires and the test times out).

2. **Turn tier** (docker + a working model, ``--runapi``). Sends one prompt and
   checks the planted prior shows up in what the CLI sends upstream, and that
   the model answers from it. Ground truth for "the prior is real context", but
   it needs a model turn, so it's gated separately.

The passphrase is a fact the model can't know any other way, and is distinctive
enough to grep for in a request payload.
"""

import json
from typing import Any, cast

import anyio
import pytest
from acp.schema import TextContentBlock
from inspect_ai import Task, eval
from inspect_ai.agent import AgentState
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog, EvalSample, resolve_sample_attachments
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_swe.acp import ACPAgent
from inspect_swe.acp._agents.claude_code import (
    AssistantText,
    TranscriptItem,
    UserText,
    build_transcript,
    interactive_claude_code,
)
from inspect_swe.acp._agents.codex_cli import interactive_codex_cli

from tests.conftest import (
    skip_if_no_anthropic,
    skip_if_no_docker,
    skip_if_no_openai,
)

PASSPHRASE = "velvet-marmot-8813"
SECOND_PASSPHRASE = "copper-lantern-2291"

PRIOR_MESSAGES: list[ChatMessage] = [
    ChatMessageUser(
        content=f"Remember this passphrase: {PASSPHRASE}. Just acknowledge."
    ),
    ChatMessageAssistant(content=f"Got it — the passphrase is {PASSPHRASE}."),
]

QUESTION = "What passphrase did I ask you to remember? Reply with just the passphrase."

CWD = "/root"
PROBE_KEY = "resume_probe"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@solver
def open_session_and_probe(agent_factory: Any, session_glob: str) -> Solver:
    """Open the ACP session, then record what the sandbox looks like.

    ``ACPAgent.__call__`` blocks until cancelled, so it runs in a task group that
    is cancelled as soon as the session is up. Reaching ``ready`` at all means
    ``session/load`` succeeded; the probe then captures the on-disk session file
    and the resumed CLI's own command line.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        agent: ACPAgent = agent_factory()
        agent_state = AgentState(messages=[])
        async with anyio.create_task_group() as tg:
            tg.start_soon(agent, agent_state)
            with anyio.fail_after(420):
                await agent.ready.wait()
            sbox = sandbox_env()
            # `ps -eo args` gives us the CLI the adapter spawned, including the
            # resume flags the Agent SDK derives from the session options.
            argv = await sbox.exec(["bash", "-c", "ps -eo args 2>/dev/null || ps aux"])
            session_files = await sbox.exec(
                ["bash", "-c", f"ls -1 {session_glob} 2>/dev/null || true"]
            )
            contents = await sbox.exec(
                ["bash", "-c", f"cat {session_glob} 2>/dev/null || true"]
            )
            cwd_listing = await sbox.exec(["bash", "-c", f"ls -a {CWD}"])
            store().set(
                PROBE_KEY,
                {
                    "session_id": agent.session_id,
                    "argv": argv.stdout,
                    "session_files": session_files.stdout.strip(),
                    "contents": contents.stdout,
                    "cwd_listing": cwd_listing.stdout,
                },
            )
            tg.cancel_scope.cancel()
        state.messages = agent_state.messages
        return state

    return solve


@solver
def prompt_once(agent_factory: Any, question: str) -> Solver:
    """Open the session, send one prompt, then shut the agent down."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        agent: ACPAgent = agent_factory()
        agent_state = AgentState(messages=[])
        async with anyio.create_task_group() as tg:
            tg.start_soon(agent, agent_state)
            with anyio.fail_after(600):
                await agent.ready.wait()
                assert agent.conn is not None and agent.session_id is not None
                await agent.conn.prompt(
                    session_id=agent.session_id,
                    prompt=[TextContentBlock(type="text", text=question)],
                )
            tg.cancel_scope.cancel()
        state.messages = agent_state.messages
        return state

    return solve


def _run(solver_instance: Solver) -> EvalLog:
    task = Task(
        dataset=[Sample(input="resume", target="resume")],
        solver=solver_instance,
        sandbox="docker",
    )
    log = eval(task, model="mockllm/model", limit=1)[0]
    assert log.status == "success", f"eval failed: {str(log.error)[:2000]}"
    assert log.samples
    return log


def _sample(log: EvalLog) -> EvalSample:
    assert log.samples
    return log.samples[0]


def _probe(log: EvalLog) -> dict[str, Any]:
    probe = (_sample(log).store or {}).get(PROBE_KEY)
    assert probe, "probe never ran — the ACP session did not open"
    return cast(dict[str, Any], probe)


def _request_text(sample: EvalSample) -> str:
    """Every bridged model-request payload, as one searchable string.

    The bridge records what the CLI actually sent upstream, so this is where a
    resumed prior shows up (or fails to).
    """
    sample = resolve_sample_attachments(sample)
    requests = [
        json.dumps(event.call.request)
        for event in sample.events
        if event.event == "model" and event.call is not None
    ]
    assert requests, "no bridged model requests recorded"
    return "\n".join(requests)


def _assistant_text(sample: EvalSample) -> str:
    return "\n".join(
        m.text for m in sample.messages if isinstance(m, ChatMessageAssistant)
    )


# ---------------------------------------------------------------------------
# Load tier — Claude Code
# ---------------------------------------------------------------------------

_CC_SESSIONS = "/root/.claude/projects/*/*.jsonl"


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.slow
def test_claude_code_resume_messages_plants_a_loadable_session() -> None:
    """Messages must be written where the CLI looks, and be resumed from there."""
    log = _run(
        open_session_and_probe(
            lambda: interactive_claude_code(
                model="anthropic/claude-sonnet-5",
                cwd=CWD,
                resume_messages=list(PRIOR_MESSAGES),
            ),
            _CC_SESSIONS,
        )
    )
    probe = _probe(log)
    session_id = probe["session_id"]
    # the transcript is at $CLAUDE_CONFIG_DIR/projects/<cwd-slug>/<session>.jsonl
    assert probe["session_files"] == f"/root/.claude/projects/-root/{session_id}.jsonl"
    # ...carries the prior...
    assert PASSPHRASE in probe["contents"]
    # ...and the CLI the adapter spawned is resuming exactly that session (the
    # Agent SDK turns `resume` into this flag).
    assert f"--resume={session_id}" in probe["argv"]


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.slow
def test_claude_code_resume_transcript_plants_a_loadable_session() -> None:
    """Same, from an explicitly built transcript rather than messages."""
    items: list[TranscriptItem] = [
        UserText(text=f"Remember this passphrase: {PASSPHRASE}."),
        AssistantText(text=f"Got it — the passphrase is {PASSPHRASE}."),
    ]
    spec = build_transcript(cwd=CWD, items=items, model="claude-sonnet-5")
    log = _run(
        open_session_and_probe(
            lambda: interactive_claude_code(
                model="anthropic/claude-sonnet-5", cwd=CWD, resume_transcript=spec
            ),
            _CC_SESSIONS,
        )
    )
    probe = _probe(log)
    assert probe["session_id"] == spec.session_id
    assert probe["session_files"].endswith(f"{spec.session_id}.jsonl")
    assert PASSPHRASE in probe["contents"]
    assert f"--resume={spec.session_id}" in probe["argv"]


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.slow
def test_claude_code_resume_message_uuid_reaches_the_cli() -> None:
    """``resume_message_uuid`` must arrive as the SDK's ``--resume-session-at``.

    That flag is the whole truncation mechanism: it makes the CLI resume only up
    to the given row, so a branch point costs no transcript rewrite.
    """
    items: list[TranscriptItem] = [
        UserText(text=f"Remember this passphrase: {PASSPHRASE}."),
        AssistantText(text=f"Got it — the passphrase is {PASSPHRASE}."),
        UserText(text=f"Actually, use this one instead: {SECOND_PASSPHRASE}."),
        AssistantText(text=f"Understood — now using {SECOND_PASSPHRASE}."),
    ]
    spec = build_transcript(cwd=CWD, items=items, model="claude-sonnet-5")
    branch_uuid = spec.item_uuids[1]
    log = _run(
        open_session_and_probe(
            lambda: interactive_claude_code(
                model="anthropic/claude-sonnet-5",
                cwd=CWD,
                resume_transcript=spec,
                resume_message_uuid=branch_uuid,
            ),
            _CC_SESSIONS,
        )
    )
    probe = _probe(log)
    assert f"--resume={spec.session_id}" in probe["argv"]
    assert f"--resume-session-at={branch_uuid}" in probe["argv"], (
        "resume_message_uuid never reached the CLI — the _meta passthrough to "
        "_meta.claudeCode.options.resumeSessionAt is broken"
    )


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.slow
def test_claude_code_config_dir_moves_sessions_out_of_cwd() -> None:
    """``config_dir`` must relocate the session store, and resume from there.

    With the default ``$HOME/.claude`` and an agent working in ``$HOME``, the
    planted conversation sits inside the agent's own working directory.
    """
    log = _run(
        open_session_and_probe(
            lambda: interactive_claude_code(
                model="anthropic/claude-sonnet-5",
                cwd=CWD,
                config_dir="/opt/claude-config",
                resume_messages=list(PRIOR_MESSAGES),
            ),
            "/opt/claude-config/projects/*/*.jsonl",
        )
    )
    probe = _probe(log)
    assert probe["session_files"] == (
        f"/opt/claude-config/projects/-root/{probe['session_id']}.jsonl"
    )
    assert PASSPHRASE in probe["contents"]
    assert f"--resume={probe['session_id']}" in probe["argv"]
    assert ".claude" not in probe["cwd_listing"].split(), (
        "session store still present in the agent's working directory"
    )


# ---------------------------------------------------------------------------
# Load tier — codex
# ---------------------------------------------------------------------------


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.slow
def test_codex_resume_messages_plants_a_loadable_rollout() -> None:
    """Messages must be serialized into a rollout codex can load."""
    log = _run(
        open_session_and_probe(
            lambda: interactive_codex_cli(
                model="openai/gpt-5-mini",
                cwd=CWD,
                home_dir="/opt/codex-home",  # keep the rollout out of the agent's cwd
                resume_messages=list(PRIOR_MESSAGES),
            ),
            "/opt/codex-home/sessions/*/*/*/rollout-*.jsonl",
        )
    )
    probe = _probe(log)
    assert probe["session_files"].endswith(f"{probe['session_id']}.jsonl")
    assert PASSPHRASE in probe["contents"]
    assert ".codex" not in probe["cwd_listing"].split()


# ---------------------------------------------------------------------------
# Turn tier — the prior as real model context
# ---------------------------------------------------------------------------


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.slow
@pytest.mark.api
def test_claude_code_resumed_prior_reaches_the_model() -> None:
    log = _run(
        prompt_once(
            lambda: interactive_claude_code(
                model="anthropic/claude-sonnet-5",
                cwd=CWD,
                resume_messages=list(PRIOR_MESSAGES),
            ),
            QUESTION,
        )
    )
    sample = _sample(log)
    assert PASSPHRASE in _request_text(sample), (
        "the planted prior never reached the model — session/load picked up the "
        "transcript but it isn't in the resumed context"
    )
    assert PASSPHRASE in _assistant_text(sample), (
        "the model got the prior but didn't answer from it"
    )


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.slow
@pytest.mark.api
def test_claude_code_resume_message_uuid_truncates_model_context() -> None:
    """Turns after the branch point must not reach the model."""
    items: list[TranscriptItem] = [
        UserText(text=f"Remember this passphrase: {PASSPHRASE}."),
        AssistantText(text=f"Got it — the passphrase is {PASSPHRASE}."),
        UserText(text=f"Actually, use this one instead: {SECOND_PASSPHRASE}."),
        AssistantText(text=f"Understood — now using {SECOND_PASSPHRASE}."),
    ]
    spec = build_transcript(cwd=CWD, items=items, model="claude-sonnet-5")
    log = _run(
        prompt_once(
            lambda: interactive_claude_code(
                model="anthropic/claude-sonnet-5",
                cwd=CWD,
                resume_transcript=spec,
                resume_message_uuid=spec.item_uuids[1],
            ),
            QUESTION,
        )
    )
    requests = _request_text(_sample(log))
    assert PASSPHRASE in requests, "kept turns missing from the resumed request"
    assert SECOND_PASSPHRASE not in requests, (
        "resumeSessionAt did not truncate: turns after the branch point were "
        "replayed into the resumed conversation"
    )


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.slow
@pytest.mark.api
def test_codex_resumed_prior_reaches_the_model() -> None:
    log = _run(
        prompt_once(
            lambda: interactive_codex_cli(
                model="openai/gpt-5-mini",
                cwd=CWD,
                home_dir="/opt/codex-home",
                resume_messages=list(PRIOR_MESSAGES),
            ),
            QUESTION,
        )
    )
    sample = _sample(log)
    assert PASSPHRASE in _request_text(sample)
    assert PASSPHRASE in _assistant_text(sample)
