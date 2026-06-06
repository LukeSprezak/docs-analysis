#!/bin/bash
set -e

echo "Running ruff check..."
uv run ruff check .

echo "Running ruff format check..."
uv run ruff format --check .

echo "Running mypy..."
uv run mypy .
