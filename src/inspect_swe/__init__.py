from ._claude_code.claude_code import claude_code

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


__all__ = ["claude_code", "__version__"]
