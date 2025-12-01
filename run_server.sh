#!/bin/bash

# ------------------------------------------------------------
# FastAPI Server Launcher
# ------------------------------------------------------------
# Runs the FastAPI backend from the root directory
# Will need to make the file executable with: chmod +x run_server.sh
# Usage: ./run_server.sh [PORT]
# ------------------------------------------------------------

PORT=${1:-8080}
RELOAD_FLAG=$2

if [[ "$RELOAD_FLAG" == "reload" ]]; then
    RELOAD="--reload"
    echo "Starting server with auto-reload on port $PORT..."
else
    RELOAD=""
    echo "Starting server on port $PORT..."
fi

uvicorn backend.main:app --host 0.0.0.0 --port $PORT $RELOAD