# backend/routers/media.py
from fastapi import APIRouter, HTTPException
import os
from pathlib import Path
from typing import List

router = APIRouter(tags=["media"])

ALLOWED_DIRECTORIES = [
    Path("/home/kody/media") # This is my media directory on the linux server. Will let users change it later with sqlite.
]

MAX_INDEXES = 50_0000  # Arbitrary limit to prevent abuse

# -----------------------------------------------------------------------------
# Simple in-memory "library" of file paths users registered this session.
# We store Path objects. This list is cleared on server restart.
# -----------------------------------------------------------------------------
file_list: List[Path] = []