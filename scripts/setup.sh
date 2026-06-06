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
    if command -v npm &> /dev/null; then
        npm install
    else
        echo "Warning: npm is not installed. Please install Node.js and npm to manage frontend dependencies locally."
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
    until [ "$(docker compose ps --filter "status=running" --format "{{.Service}}" | grep -w db)" ] && [ "$(docker inspect -f {{.State.Health.Status}} $(docker compose ps -q db))" == "healthy" ] || [ $COUNT -eq $MAX_RETRIES ]; do
        echo "Waiting for db to be healthy... ($COUNT/$MAX_RETRIES)"
        sleep 2
        COUNT=$((COUNT+1))
    done
else
    echo "Skipping Docker-based database setup. Ensure you have a local PostgreSQL instance running if needed."
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
