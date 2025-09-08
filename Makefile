.PHONY: hooks
hooks:
	pre-commit install

.PHONY: check
check:
	ruff check --fix
	ruff format
	mypy src tests

.PHONY: test
test:
	pytest

.PHONY: sync
sync:
	uv sync --all-groups --all-extras