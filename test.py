# backend/routers/media.py
# -----------------------------------------------------------------------------
# A tiny media router that:
#   1) Stores ABSOLUTE file paths in an in-memory list (POST /register_path)
#   2) Shows the list (GET /list_paths)
#   3) Streams a selected file with HTTP Range support (GET /stream_by_index/{index})
#   4) (Optional) Live-transcodes a file to MP4 using ffmpeg (GET /transcode_by_index/{index})
#
# Notes:
# - We NEVER upload file contents; we just remember local paths and read them.
# - The list is in-memory, so restarting the server clears it.
# - We restrict readable files to ALLOWED_DIRS for safety.
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import List
import mimetypes
import subprocess
import os

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse

router = APIRouter(tags=["media"])

# -----------------------------------------------------------------------------
# Simple in-memory "library" of file paths users registered this session.
# We store Path objects. This list is cleared on server restart.
# -----------------------------------------------------------------------------
file_list: List[Path] = []

# -----------------------------------------------------------------------------
# SECURITY: Only allow reading files INSIDE these directories.
# Edit this to match your machine. Keep this tight!
# If a path isn't under one of these folders, we reject it with 403.
# -----------------------------------------------------------------------------
ALLOWED_DIRS = [
    Path("/home/kody/media"),
    Path.home() / "Videos",
    Path.home() / "Music",
    # Examples you could add:
    # Path("D:/Media"),         # Windows drive
    # Path("/mnt/media"),       # Mounted NAS
]

# --- NEW: media type config ---------------------------------------------------
MEDIA_EXTS = {
    # common video
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi",
    # common audio
    ".mp3", ".aac", ".m4a", ".flac", ".wav", ".ogg", ".opus",
}

# Optional: cap how much we index on startup (protects against huge trees)
MAX_INDEX = 50_000

from threading import Lock
_index_lock = Lock()

# Normalize and keep only the dirs that actually exist on this machine.
ALLOWED_DIRS = [p.resolve() for p in ALLOWED_DIRS if p.exists()]

# --- NEW: helper: is this file "likely playable"? ----------------------------
def is_likely_media(p: Path) -> bool:
    # Fast path: extension check
    if p.suffix.lower() in MEDIA_EXTS:
        return True
    # Fallback: mimetypes for odd extensions
    mime, _ = mimetypes.guess_type(str(p))
    return bool(mime and (mime.startswith("video/") or mime.startswith("audio/")))

# -----------------------------------------------------------------------------
# Helper: check that a given ABSOLUTE path is a file and lives under ALLOWED_DIRS
# -----------------------------------------------------------------------------
def ensure_allowed_file(abs_path: Path) -> Path:
    """
    Make sure:
      - 'abs_path' points to a real file
      - it lives under one of ALLOWED_DIRS
    Returns the resolved Path if OK; else raises HTTPException.
    """
    resolved = abs_path.resolve()

    # Must exist and be a file (not a folder)
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Must be under one of our allowed directories
    for allowed_dir in ALLOWED_DIRS:
        try:
            resolved.relative_to(allowed_dir)  # raises ValueError if outside
            return resolved
        except ValueError:
            continue

    raise HTTPException(status_code=403, detail="Path is outside allowed roots")

# -----------------------------------------------------------------------------
# Helper: yield a chunked byte range from an open file.
# Used for HTTP Range (seek/scrub) support.
# -----------------------------------------------------------------------------
def read_range_chunks(file_obj, start: int, end: int, chunk_size: int = 1024 * 1024):
    """
    Yield bytes from [start, end] inclusive, in chunk_size pieces.
    """
    file_obj.seek(start)
    remaining = end - start + 1
    while remaining > 0:
        data = file_obj.read(min(chunk_size, remaining))
        if not data:
            break
        yield data
        remaining -= len(data)

# --- NEW: directory scanner ---------------------------------------------------
def scan_allowed_dirs():
    """
    Walk ALLOWED_DIRS, collect playable files, enforce safety via ensure_allowed_file,
    and return a de-duplicated list of Paths.
    """
    discoverd: list[Path] = []
    seen = set()

    count = 0
    for root in ALLOWED_DIRS:
        # Defensive: skip if the dir dissapears
        if not root.exists():
            continue
        
        for dirpath, dirnames, filenames in os.walk(root):
            # (Optional) prune hidden folders for speed/noise
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            for name in filenames:
                # quick extension check before stat/resolve
                p = Path(dirpath) / name
                if not is_likely_media(p):
                    continue

                try:
                    safe = ensure_allowed_file(p)
                except HTTPException:
                    continue  # skip unsafe files

                rp = safe.resolve()
                if rp in seen:
                    continue # skip duplicates

                try:
                    if rp.stat().st_size <= 0:
                        continue #skip empty files
                except OSError:
                    continue # skip files we can't stat

                discoverd.append(rp)
                seen.add(rp)
                count += 1
                if count >= MAX_INDEX:
                    return discoverd
    return discoverd

@router.on_event("startup")
def _seed_library_on_startup():
    if not ALLOWED_DIRS:
        print("[media.py] Startup: no ALLOWED_DIRS; skipping scan.")
        return
    with _index_lock:
        discovered = scan_allowed_dirs()
        # Replace (not extend) so a restart gives a clean read
        file_list.clear()
        file_list.extend(discovered)
    print(f"[media.py] Startup scan indexed {len(file_list)} items.")


# --- NEW: manual rescan endpoint ---------------------------------------------
@router.post("/rescan")
def rescan_library():
    """
    Manually re-scan ALLOWED_DIRS and replace the in-memory library.
    Useful if you add/remove files while the server is running.
    """
    if not ALLOWED_DIRS:
        raise HTTPException(status_code=400, detail="No ALLOWED_DIRS configured.")
    with _index_lock:
        discovered = scan_allowed_dirs()
        file_list.clear()
        file_list.extend(discovered)
    return {"message": "Rescan complete", "count": len(file_list)}


# -----------------------------------------------------------------------------
# POST /api/register_path?path=/absolute/path/to/file
# - Validate the path and remember it in file_list
# -----------------------------------------------------------------------------
@router.post("/register_path")
def register_path(
    path: str = Query(..., description="Absolute path to a local media file")
):
    """
    Register an ABSOLUTE file path into our in-memory list.
    We don't upload content; we just remember where the file is.
    """
    abs_path = Path(path)

    if not abs_path.is_absolute():
        raise HTTPException(status_code=400, detail="Path must be absolute")

    allowed_path = ensure_allowed_file(abs_path)
    file_list.append(allowed_path)

    return {
        "message": f"Registered: {allowed_path}",
        "index": len(file_list) - 1,
    }

# -----------------------------------------------------------------------------
# GET /api/list_paths
# - Return the current list of registered file paths as strings
# -----------------------------------------------------------------------------
@router.get("/list_paths")
def list_paths():
    """
    Return the list of registered absolute file paths as strings.
    Note: This list resets when the server restarts.
    """
    return JSONResponse([str(p) for p in file_list])

# -----------------------------------------------------------------------------
# GET /api/stream_by_index/{index}
# - Stream the file at file_list[index] with HTTP Range support
#   so the browser can seek/scrub properly.
# -----------------------------------------------------------------------------
@router.get("/stream_by_index/{index}")
def stream_by_index(index: int, request: Request):
    """
    Stream a registered file with proper Range support (206 Partial Content).
    Works best with browser-supported formats (e.g., .mp4 H.264/AAC).
    """
    # Basic index safety
    if index < 0 or index >= len(file_list):
        raise HTTPException(status_code=404, detail="Invalid index")

    path_obj = file_list[index]
    total_size = path_obj.stat().st_size

    # Tell the browser what kind of file this is (video/mp4, audio/mpeg, etc.)
    mime, _ = mimetypes.guess_type(str(path_obj))
    if not mime:
        mime = "application/octet-stream"

    # We always advertise that we support byte ranges
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",  # don't cache while iterating locally
    }

    # If the browser asked for a specific byte range (for seeking/scrubbing)
    range_header = request.headers.get("range")
    if range_header:
        # Typical format: "Range: bytes=START-END" (END optional)
        units, _, byte_range = range_header.partition("=")
        if units.strip().lower() != "bytes":
            # Reject any units we don't support
            return PlainTextResponse("Invalid range unit", status_code=416)

        start_str, _, end_str = byte_range.strip().partition("-")
        try:
            if start_str == "":
                # Suffix form: "bytes=-N" → last N bytes
                suffix_len = int(end_str)
                start = max(total_size - suffix_len, 0)
                end = total_size - 1
            else:
                start = int(start_str)
                end = int(end_str) if end_str else (total_size - 1)
        except ValueError:
            return PlainTextResponse("Invalid range", status_code=416)

        # Sanity checks on computed range
        if start > end or start >= total_size:
            return PlainTextResponse("Invalid range", status_code=416)

        content_len = end - start + 1
        headers.update({
            "Content-Range": f"bytes {start}-{end}/{total_size}",
            "Content-Length": str(content_len),
        })

        # Stream only the requested byte range
        def range_stream():
            with open(path_obj, "rb") as f:
                yield from read_range_chunks(f, start, end)

        return StreamingResponse(
            range_stream(),
            status_code=206,         # Partial Content
            media_type=mime,
            headers=headers,
        )

    # No Range header → stream the whole file in chunks
    headers["Content-Length"] = str(total_size)

    def full_stream():
        with open(path_obj, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        full_stream(),
        media_type=mime,
        headers=headers,
    )

# -----------------------------------------------------------------------------
# GET /api/transcode_by_index/{index}
# - Live transcode to MP4 (H.264 + AAC) using ffmpeg and stream it progressively.
# - This is a minimal "works now" version. It does NOT support Range,
#   so scrubbing may be limited until enough buffer is loaded.
# -----------------------------------------------------------------------------
@router.get("/transcode_by_index/{index}")
def transcode_by_index(index: int):
    """
    Live-transcode the selected file to MP4 (H.264/AAC) and stream it.
    Use this for formats browsers can't play (e.g., .wmv).
    Requires ffmpeg to be installed on the server.

    NOTE: No Range support here. For fully seekable live streaming,
    you'd move to HLS/DASH or add a more advanced range-aware pipeline.
    """
    # Basic index safety
    if index < 0 or index >= len(file_list):
        raise HTTPException(status_code=404, detail="Invalid index")

    src_path = file_list[index]
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Minimal ffmpeg pipeline:
    # - Read input file
    # - Encode video to H.264 and audio to AAC
    # - MP4 container
    # - Write to stdout (pipe:1) so we can stream it out
    # - +frag_keyframe+empty_moov+faststart → lets playback begin ASAP
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", str(src_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+frag_keyframe+empty_moov+faststart",
        "-f", "mp4",
        "pipe:1",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,   # We will read encoded MP4 bytes from here
            stderr=subprocess.PIPE,   # Keep stderr to avoid blocking on noisy logs
        )
    except FileNotFoundError:
        # ffmpeg isn't installed or not in PATH
        raise HTTPException(status_code=500, detail="ffmpeg not found. Install it on the server.")

    # Yield encoded MP4 bytes as they are produced by ffmpeg
    def transcode_stream():
        try:
            while True:
                data = proc.stdout.read(64 * 1024)  # 64 KB chunks
                if not data:
                    break
                yield data
        finally:
            # If the client disconnects early, make sure ffmpeg is stopped
            if proc and proc.poll() is None:
                proc.kill()

    # We omit Content-Length and Accept-Ranges because this is live output
    headers = {
        "Cache-Control": "no-store",
    }

    return StreamingResponse(
        transcode_stream(),
        media_type="video/mp4",
        headers=headers,
    )

# (Optional) Log the allowed dirs at import time so it's obvious what's active.
if not ALLOWED_DIRS:
    print("[media.py] WARNING: ALLOWED_DIRS is empty; all register_path calls will 403.")
else:
    print("[media.py] ALLOWED_DIRS:", [str(p) for p in ALLOWED_DIRS])
