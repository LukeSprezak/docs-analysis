#!/bin/bash
set -e

echo "Running ruff check with fix..."
uv run ruff check --fix .

echo "Running ruff format..."
uv run ruff format .

echo "Running mypy..."
uv run mypy .
