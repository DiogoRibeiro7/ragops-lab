.PHONY: install lint typecheck test format run

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

run:
	poetry run python -m ragops_lab.cli
