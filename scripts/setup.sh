#!/bin/bash
set -e

echo "Starting project configuration..."

if [ ! -f .env ]; then
    echo "Creating a .env file from .env.example..."
    cp .env.example .env
else
    echo "The .env file already exists."
fi

FORCE_BUILD=false
for arg in "$@"; do
    if [ "$arg" == "--build" ]; then
        FORCE_BUILD=true
    fi
done

echo "Installing/updating dependencies with uv..."
if command -v uv &> /dev/null; then
    uv sync
else
    echo "Warning: uv is not installed. Please install it to manage dependencies locally."
fi

echo "Installing frontend dependencies..."
if [ -d "client" ]; then
    cd client
    export COREPACK_ENABLE_DOWNLOAD_PROMPT=0
    if command -v corepack &> /dev/null; then
        corepack enable
        yarn install
    elif command -v yarn &> /dev/null; then
        echo "Warning: corepack not found — falling back to the yarn on PATH, which may not be the pinned 4.x."
        yarn install
    else
        echo "Warning: no package manager found. Install Node.js, then: npm install -g corepack && corepack enable"
    fi
    cd ..
else
    echo "Warning: client directory not found."
fi

USE_DOCKER=true
if ! command -v docker &> /dev/null; then
    echo "Warning: docker is not installed. Skipping Docker setup."
    USE_DOCKER=false
elif ! docker info &> /dev/null; then
    echo "Warning: Docker daemon is not running. Skipping Docker setup."
    USE_DOCKER=false
fi

if [ "$USE_DOCKER" = true ]; then
    echo "Checking for Docker images..."
    API_IMAGE=$(docker images -q docsanalysis-api:latest 2> /dev/null || echo "")

    if [ -z "$API_IMAGE" ] || [ "$FORCE_BUILD" = true ]; then
        echo "Building Docker images..."
        docker compose up -d --build
    else
        echo "The images already exist. Launching containers without rebuilding..."
        docker compose up -d
    fi

    echo "Waiting for the database to be ready..."
    MAX_RETRIES=30
    COUNT=0

    until DB_ID=$(docker compose ps -q db) \
        && [ -n "$DB_ID" ] \
        && [ "$(docker inspect -f '{{.State.Health.Status}}' "$DB_ID" 2>/dev/null)" == "healthy" ]; do
        if [ "$COUNT" -ge "$MAX_RETRIES" ]; then
            echo "Error: db did not become healthy within $((MAX_RETRIES * 2))s." >&2
            docker compose logs db --tail=20 >&2
            exit 1
        fi
        echo "Waiting for db to be healthy... ($COUNT/$MAX_RETRIES)"
        sleep 2
        COUNT=$((COUNT + 1))
    done

    echo "Applying database migrations..."
    docker compose exec -T api uv run alembic upgrade head
else
    echo "Skipping Docker-based database setup. Ensure you have a local PostgreSQL instance running if needed."
    echo "Applying database migrations..."
    echo "Note: this needs POSTGRES_HOST/PORT in .env to point at your local instance (not 'db')."
    uv run alembic upgrade head
fi

echo "Running linting..."
chmod +x ./scripts/lint.sh
./scripts/lint.sh

echo "Running tests..."
if [ -f ./scripts/tests.sh ]; then
    chmod +x ./scripts/tests.sh
    ./scripts/tests.sh
else
    echo "Tests script not found, running pytest via uv..."
    uv run pytest
fi

echo "The setup was successful!"
echo "The app is available at: http://localhost:8001/docs"
echo "Frontend is available at: http://localhost:3000"
