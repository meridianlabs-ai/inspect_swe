from logging import getLogger

logger = getLogger(__file__)


def trace(message: str) -> None:
    logger.info(f"[Inspect SWE] {message}")
