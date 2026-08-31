import base64
import struct
import zlib

from inspect_ai import Task, task
from inspect_ai.agent import run
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import SandboxEnvironmentType
from inspect_swe import codex_cli


def solid_color_image(r: int, g: int, b: int, size: int = 64) -> ContentImage:
    """A solid-color PNG as image content (stdlib-only, no PIL)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = struct.pack(">I", len(data)) + tag + data
        return payload + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes([r, g, b]) * size for _ in range(size))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return ContentImage(image=f"data:image/png;base64,{base64.b64encode(png).decode()}")


@solver
def image_input_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        agent = codex_cli(system_prompt="Answer simple questions concisely")

        # first run: image in the sample input (codex exec --image)
        agent_state = await run(agent, state.messages)

        # second run: image in a follow-up message (codex exec resume --image)
        agent_state.messages.append(
            ChatMessageUser(
                content=[
                    ContentText(
                        text="Here is a second image. What solid color is it? "
                        "Answer with just the color name."
                    ),
                    solid_color_image(0, 200, 0),
                ]
            )
        )
        agent_state = await run(agent, agent_state)

        # transfer state and return
        state.messages = agent_state.messages
        state.output = agent_state.output
        return state

    return solve


@task
def image_input(sandbox: SandboxEnvironmentType | None = "docker") -> Task:
    return Task(
        dataset=[
            Sample(
                input=[
                    ChatMessageUser(
                        content=[
                            ContentText(
                                text="What solid color is the attached image? "
                                "Answer with just the color name."
                            ),
                            solid_color_image(255, 0, 255),
                        ]
                    )
                ]
            )
        ],
        solver=image_input_solver(),
        sandbox=sandbox,
    )
