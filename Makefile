.EXPORT_ALL_VARIABLES:
.PHONY: setup venv install install-dev install-poetry install-dev-deps pre-commit clean check-lint test

PYTHON_VERSION = python3.10
CUDNN_VERSION = 8.9.5.29
CUDA_VERSION = cuda12.2
DISTRO = ubuntu2204
ARCH = x86_64
CUDA_KEYRING_FILE = cuda-keyring_1.1-1_all.deb

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

install-cuda-wsl:
	@echo "Install NVIDIA CUDA Toolkit"
	sudo apt-get install nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc

	@echo "Install the new cuda-keyring package"
	wget https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb
	sudo dpkg -i {CUDA_KEYRING_FILE}
	rm -f {CUDA_KEYRING_FILE}

	@echo "Install cuDNN"
	sudo apt-get install libcudnn8=${CUDNN_VERSION}-1+${CUDA_VERSION}
	sudo apt-get install libcudnn8-dev=${CUDNN_VERSION}-1+${CUDA_VERSION}
	sudo apt-get install libcudnn8-samples=${CUDNN_VERSION}-1+${CUDA_VERSION}
