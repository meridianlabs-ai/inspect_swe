import os


def join_path(base: str, path: str) -> str:
    return os.path.join(base, path).replace("\\", "/")
