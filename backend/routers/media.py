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
MEDIA_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv"} # Base media extension

# -----------------------------------------------------------------------------
# Simple in-memory "library" of file paths users registered this session.
# We store Path objects. This list is cleared on server restart.
# -----------------------------------------------------------------------------
file_list: List[Path] = []


# -----------------------------------------------------------------------------
# Im writing a simple helper function to check if a path contains any readable media files.
# -----------------------------------------------------------------------------
def _contains_media_files(directory: Path)  -> bool:
    # Check each file in the directory for media extensions
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS:
            return True
        

@router.on_event("startup")
def startup_event():
    """
    On startup, ensure allowed directories exist.
    Check and see if there are any media files in those directories.
    Make sure that those files are more than 0 bytes.
    Store those files in the in-memory file_list.
    """
    for directory in ALLOWED_DIRECTORIES:
        if not directory.exists() or not directory.is_dir():
            raise RuntimeError(f"Allowed directory does not exist or is not a directory: {directory}")

        # Run helper function to check if the directory contains any media file before scanning.
        _contains_media_files(directory)

        # Scan the directory for media files
        for item in directory.rglob('*'):
            if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS:
                file_list.append(item)