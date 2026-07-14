.PHONY: hooks
hooks:
	pre-commit install

.PHONY: check
check:
	ruff format
	ruff check --fix
	mypy

.PHONY: test
test:
	pytest

.PHONY: sync
sync:
	uv sync --all-groups --all-extras