# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    RM := rmdir /s /q
else
    DETECTED_OS := $(shell uname -s)
    RM := rm -rf
endif

# Variables
NODE_MODULES := node_modules
PACKAGE_LOCK := package-lock.json

.PHONY: help all install dev build preview clean

help:
	@echo "Available commands:"
	@echo "  make all        - Install dependencies and run dev server"
	@echo "  make install    - Install npm dependencies"
	@echo "  make dev        - Run development server"
	@echo "  make build      - Build for production"
	@echo "  make preview    - Preview production build"
	@echo "  make clean      - Remove node_modules and build files"
	@echo ""
	@echo "Detected OS: $(DETECTED_OS)"

# Do everything: install and run
all: install dev

# Install dependencies
install:
	@echo "Installing dependencies..."
	npm install
	@echo "Dependencies installed!"

# Run development server
dev:
	@echo "Starting development server..."
	npm run dev

# Build for production
build:
	@echo "Building for production..."
	npm run build
	@echo "Build complete!"

# Preview production build
preview:
	@echo "Previewing production build..."
	npm run preview

# Clean up
clean:
	@echo "Cleaning up..."
	$(RM) $(NODE_MODULES)
	$(RM) dist
	@echo "Cleanup complete!"