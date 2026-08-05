.PHONY: install lint typecheck test format check pre-commit run

install:
	poetry install --with dev

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

test:
	poetry run pytest -q

format:
	poetry run ruff format .

check: lint typecheck test

pre-commit:
	poetry run pre-commit run --all-files

run:
	poetry run python -m ragops_lab.cli
