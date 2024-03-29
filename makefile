sources = thermal

.PHONY: test format lint unittest coverage pre-commit clean
test: format lint unittest

format:
	isort $(sources) tests
	black $(sources) tests

lint:
	poetry run flake8 $(sources) tests
	poetry run mypy $(sources) tests

unittest:
	pytest

coverage:
	poetry run pytest --cov=$(sources) --cov-branch --cov-report=term-missing tests

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .mypy_cache .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage

html:
	poetry run sphinx-build -b html docs docs/_build/html

version:
	poetry run bump2version patch
