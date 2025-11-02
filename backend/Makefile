# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    VENV_BIN := venv\Scripts
    PYTHON := python
    RM := rmdir /s /q
    MKDIR := mkdir
else
    DETECTED_OS := $(shell uname -s)
    VENV_BIN := venv/bin
    PYTHON := python3
    RM := rm -rf
    MKDIR := mkdir -p
endif

# Variables
VENV := venv
REQUIREMENTS := requirements.txt
APP := app:app
HOST := 127.0.0.1
PORT := 8000

.PHONY: help all install venv dev clean

help:
	@echo "Available commands:"
	@echo "  make all        - Install dependencies and run server"
	@echo "  make install    - Create venv and install dependencies"
	@echo "  make dev        - Run development server"
	@echo "  make clean      - Remove venv and cache files"
	@echo ""
	@echo "Detected OS: $(DETECTED_OS)"

# Do everything: install and run
all: install dev

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created!"

# Install dependencies
install: venv
	@echo "Installing dependencies..."
ifeq ($(OS),Windows_NT)
	$(VENV_BIN)\pip install -r $(REQUIREMENTS)
else
	$(VENV_BIN)/pip install -r $(REQUIREMENTS)
endif
	@echo "Dependencies installed!"

# Run development server
dev:
	@echo "Starting development server on $(HOST):$(PORT)..."
ifeq ($(OS),Windows_NT)
	$(VENV_BIN)\python -m uvicorn $(APP) --reload --host $(HOST) --port $(PORT)
else
	$(VENV_BIN)/python -m uvicorn $(APP) --reload --host $(HOST) --port $(PORT)
endif

# Clean up
clean:
	@echo "Cleaning up..."
	$(RM) $(VENV)
	@echo "Cleanup complete!"