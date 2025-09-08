from logging import getLogger

from inspect_ai.util import trace_message

logger = getLogger(__file__)


def trace(message: str) -> None:
    logger.setLevel("TRACE")
    trace_message(logger, category="Inspect SWE", message=message)
