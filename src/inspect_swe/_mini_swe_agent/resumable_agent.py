"""ResumableAgent for mini-swe-agent v2 multi-turn support.

Written to sandbox at /var/tmp/resumable_agent.py and loaded via:
    mini --agent-class resumable_agent.ResumableAgent

Env vars:
    MSWEA_RESUME: "true" to resume from prior trajectory
"""

from __future__ import annotations

import json
import os
from typing import Any

from minisweagent.agents.default import DefaultAgent  # type: ignore
from minisweagent.exceptions import InterruptAgentFlow  # type: ignore

# Trajectory format version used by mini-swe-agent v2.x
# https://mini-swe-agent.com/latest/usage/output_files/#trajectory-files-trajjson
_EXPECTED_FORMAT = "mini-swe-agent-1.1"


def _trajectory_to_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate format and return messages from a v2 trajectory.

    Keeps all fields (role, content, tool_calls, tool_call_id, extra).
    Skips any entries that lack a role.
    """
    traj_format = data.get("trajectory_format", "unknown")
    if traj_format != _EXPECTED_FORMAT:
        raise RuntimeError(
            f"Trajectory format '{traj_format}' is not supported. "
            f"Expected '{_EXPECTED_FORMAT}'. The trajectory may have been "
            f"created by a different version of mini-swe-agent."
        )
    return [dict(entry) for entry in data.get("messages", []) if entry.get("role")]


def _fix_dangling_tool_calls(messages: list[dict[str, Any]]) -> None:
    """Remove tool calls from assistant message if they are unanswered tool_calls."""
    if not messages:
        return

    last = messages[-1]
    if last.get("role") != "assistant" or "tool_calls" not in last:
        return

    call_ids = {tc.get("id") for tc in last.get("tool_calls", []) if tc.get("id")}
    answered_ids = {
        msg["tool_call_id"]
        for msg in messages
        if msg.get("role") == "tool" and "tool_call_id" in msg
    }
    if call_ids - answered_ids:
        # Strip tool_calls but keep text content for message alternation
        del last["tool_calls"]
        # If there's no meaningful content left, replace with minimal text
        if not last.get("content"):
            last["content"] = "Task completed."


def _load_trajectory_data(path: str) -> dict[str, Any]:
    """Read and validate a trajectory JSON file.

    Returns the parsed trajectory dict on success, raises RuntimeError
    with a descriptive message on any failure.
    """
    try:
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Cannot resume: trajectory file not found at {path}. "
            f"The first run may not have completed successfully."
        ) from e
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Cannot resume: trajectory file at {path} contains "
            f"invalid JSON (line {e.lineno}, col {e.colno}). "
            f"The file may have been partially written."
        ) from e

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Cannot resume: trajectory file at {path} contains "
            f"{type(data).__name__} instead of expected JSON object."
        )
    return data


class ResumableAgent(DefaultAgent):  # type: ignore[misc]
    def run(self, task: str = "", **kwargs: Any) -> dict[str, Any]:
        # First attempt executes normally
        if os.environ.get("MSWEA_RESUME", "false") != "true":
            return super().run(task=task, **kwargs)  # type: ignore[no-any-return]

        # 2 or more attempts load state from prior trajectory
        data = _load_trajectory_data(str(self.config.output_path))
        self.messages = _trajectory_to_messages(data)

        # Restore cost/call (v2 query() checks self.model.cost, not self.cost)
        info = data.get("info", {})
        stats = info.get("model_stats", {})
        self.cost = stats.get("instance_cost", 0.0)
        self.n_calls = stats.get("api_calls", 0)
        self.model.cost = self.cost
        self.model.n_calls = self.n_calls

        # Strip exit messages so the agent can continue
        while self.messages and self.messages[-1].get("role") == "exit":
            self.messages.pop()
        _fix_dangling_tool_calls(self.messages)

        # Append the new task with a short exit reminder to help the agent exit.
        resume_task = (
            f"{task}\n\nWhen you are done, submit by running: "
            "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`"
        )
        self.add_messages(self.model.format_message(role="user", content=resume_task))

        # Below loop identical to original
        # https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py#L85
        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                try:
                    self.save(self.config.output_path)
                except Exception:
                    pass
            if self.messages[-1].get("role") == "exit":
                break

        return self.messages[-1].get("extra", {}) if self.messages else {}
