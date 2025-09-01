from ._claude_code.claude_code import ClaudeCodeOptions, claude_code
from ._tools.download import download_agent_binary
from ._util.sandbox import SandboxPlatform

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


__all__ = [
    "claude_code",
    "ClaudeCodeOptions",
    "download_agent_binary",
    "SandboxPlatform",
    "__version__",
]
