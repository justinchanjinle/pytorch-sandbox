.EXPORT_ALL_VARIABLES:
.PHONY: setup venv install install-dev install-poetry install-dev-deps pre-commit clean check-lint test

PYTHON_VERSION = python3.10

setup: install-dev-deps venv install pre-commit

venv:
	@echo "Creating .venv..."
	poetry env use ${PYTHON_VERSION}

install:
	@echo "Installing dependencies..."
	poetry install

install-dev:
	@echo "Installing Python dev dependencies..."
	poetry install --only dev

install-poetry:
	@echo "Installing dev dependencies..."
	curl -sSL https://install.python-poetry.org | ${PYTHON_VERSION} -

install-dev-deps:
	@echo "Installing OS dev dependencies..."
	sudo apt-get update -y
	sudo apt-get install -y aspell

pre-commit:
	@echo "Setting up pre-commit..."
	poetry run pre-commit install
	poetry run pre-commit autoupdate

clean:
	if exist .\\.git\\hooks ( rmdir .\\.git\\hooks /q /s )
	if exist .\\.venv\\ ( rmdir .\\.venv /q /s )
	if exist poetry.lock ( del poetry.lock /q /s )

check-lint:
	@set -e

	@echo "Running black check"
	poetry run black --check .

	@echo "Running flake8 check"
	poetry run flake8 .

	@echo "Running mypy check"
	poetry run mypy .

	@echo "Running spelling check"
	poetry run pyspelling

test:
	@echo "Running unit tests"
	poetry run pytest tests
