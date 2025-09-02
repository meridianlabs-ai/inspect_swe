.PHONY: hooks
hooks:
	uv run pre-commit install

.PHONY: check
check:
	uv run ruff check --fix
	uv run ruff format
	uv run mypy src tests

.PHONY: test
test:
	uv run pytest

.PHONY: sync
sync:
	uv sync --all-groups --all-extras