# backend/main.py
# -----------------------------------------------------------------------------
# This file ONLY wires the app together:
#  - creates the FastAPI app
#  - includes routers from backend/routers/*
#  - serves the static index.html
# No business logic lives here.
# -----------------------------------------------------------------------------

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from starlette.staticfiles import StaticFiles

# Import routers (the logic) from the routers package
from .routers import media_router
# Import the DB init function
from backend.db import init_db

app = FastAPI(title="Mini Local Plex")

# --- Database initialization -------------------------------------------------
@app.on_event("startup")
def startup_event():
    """
    Ensures the SQLite database and schema exist on startup.
    Runs backend/sql/init.sql automatically.
    """
    init_db()
    print("✅ Database initialized.")

# --- Static files (serve /static and / -> index.html) -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]     # project/
STATIC_DIR = BASE_DIR / "static"

# /static/* will serve files from the static directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# GET /  -> /static/index.html
@app.get("/", include_in_schema=False)
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return RedirectResponse(url="/docs")  # fallback if index.html missing

# --- Routers (ALL app logic lives under backend/routers) ----------------------
# We use a common prefix /api so the frontend can call /api/...
app.include_router(media_router, prefix="/api")
