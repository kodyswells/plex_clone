#!/bin/bash

# ------------------------------------------------------------
# FastAPI Server Launcher
# ------------------------------------------------------------
# Runs the FastAPI backend from the root directory
# Will need to make the file executable with: chmod +x run_server.sh
# Usage: ./run_server.sh [PORT]
# ------------------------------------------------------------

PORT=${1:-8000}

uvicorn backend.main:app --host 0.0.0.0 --port $PORT --reload