# backend/routers/media.py
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, RedirectResponse
from pathlib import Path
from typing import List, Iterator, Optional
from starlette.concurrency import iterate_in_threadpool
from fastapi import BackgroundTasks

import os
import mimetypes
import json, shutil, subprocess



router = APIRouter(tags=["media"])

ALLOWED_DIRECTORIES = [
    Path("/home/kody/media") # This is my media directory on the linux server. Will let users change it later with sqlite. This is just for testing.
]

MAX_INDEXES = 50_0000  # Arbitrary limit to prevent abuse
MEDIA_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv"} # Base media extension

# -----------------------------------------------------------------------------
# Simple in-memory "library" of file paths users registered this session.
# We store Path objects. This list is cleared on server restart.
# -----------------------------------------------------------------------------
file_list: List[Path] = []

# -------------------------------------------------------------------
# CONFIG: tune chunk size for disk->socket streaming
#  - Larger chunks reduce syscall overhead but increase latency/memory per client.
#  - 64 KiB is a simple, safe default.
# -------------------------------------------------------------------
STREAM_CHUNK_SIZE = 64 * 1024  # 64 KiB


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
        if _contains_media_files(directory):
            directory_file_count = 0

            # Scan the directory for media files. Using rglob to include subdirectories.
            for item in directory.rglob('*'):
                # Check to see if we have reached max indexes
                if len(file_list) >= MAX_INDEXES:
                    break
                #If not, check if the item is a file with a valid media extension and size > 0 bytes.
                elif item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS and item.stat().st_size > 0:
                    directory_file_count += 1
                    file_list.append(item)
            print(f"Indexed {directory_file_count} media files from {directory}")

# ---------------------------------------------------------
# Ensure we have ffmpeg/ffprobe available on the system
# ---------------------------------------------------------
def _ensure_ffmpeg_installed() -> None:
    """
    Raises a clear HTTP 500 if ffmpeg/ffprobe are not on PATH.
    This prevents confusing 'file not found' errors later.
    """
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg/ffprobe not found on server PATH. Install ffmpeg to enable transcoding."
        )

# ---------------------------------------------------------
# Probe a file with ffprobe to discover container/codecs
# ---------------------------------------------------------
def _ffprobe_streams(path: Path) -> dict:
    """
    Runs ffprobe and returns a dict with 'container', 'video_codec', and 'audio_codec'.
    If probing fails (corrupt file, permissions), raise a 415 (unsupported media).
    """
    try:
        # -v error keeps logs clean; -print_format json for easy parsing.
        # -show_streams prints codec info. We also capture container via -show_format.
        cmd = [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_streams", "-show_format",
            str(path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        # Could not probe → likely not a valid media file or unreadable
        raise HTTPException(status_code=415, detail=f"Unable to probe media: {e.output.decode(errors='ignore')[:200]}")

    container = (data.get("format") or {}).get("format_name", "")  # e.g., "mov,mp4,m4a,3gp,3g2,mj2"
    vcodec = None
    acodec = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and vcodec is None:
            vcodec = s.get("codec_name")  # e.g., "h264", "hevc", "vp9"
        if s.get("codec_type") == "audio" and acodec is None:
            acodec = s.get("codec_name")  # e.g., "aac", "opus", "dts"
    return {
        "container": container,
        "video_codec": vcodec,
        "audio_codec": acodec,
    }


# ---------------------------------------------------------
# Decide if we can direct-stream to a <video> tag as-is
# ---------------------------------------------------------
def _is_browser_native_playable(path: Path) -> bool:
    """
    Conservative MVP rule:
      - Container: must be MP4-family
      - Video: h264
      - Audio: aac (common on the web)
    If any of these don't match, we’ll transcode.
    """
    info = _ffprobe_streams(path)

    # Container heuristics: ffprobe format_name can be a comma-separated list.
    container = (info["container"] or "").lower()
    mp4_like = any(name in container for name in ("mp4", "mov", "m4a", "3gp", "3g2", "mj2"))

    v_ok = (info["video_codec"] or "").lower() == "h264"
    a_ok = (info["audio_codec"] or "").lower() == "aac"

    return mp4_like and v_ok and a_ok


# -----------------------------------------------------------------------------
# Build ffmpeg command for transcoding
# -----------------------------------------------------------------------------
def _build_ffmpeg_cmd(path: Path, height: int|None, v_bitrate: str, a_bitrate: str, start: float|None) -> list[str]:
    """
    Build an ffmpeg command that:
      - Optionally seeks to `start` seconds
      - (Optionally) scales to target height (maintaining aspect)
      - Encodes H.264 + AAC
      - Emits a streamable fragmented MP4 to stdout
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start and start > 0:
        # Place -ss before -i to do fast (keyframe) seek for startup speed
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]

    if height:
        cmd += ["-vf", f"scale=-2:{height}"]  # -2 keeps width divisible by 2

    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "film",
        "-b:v", v_bitrate, "-maxrate", v_bitrate, "-bufsize", str(int(int(v_bitrate[:-1]) * 2)) + "k",
        "-c:a", "aac", "-b:a", a_bitrate,
        "-movflags", "frag_keyframe+empty_moov",  # enable progressive playback
        "-f", "mp4", "pipe:1"                     # write MP4 to stdout
    ]
    return cmd

# -----------------------------------------------------------------------------
# Endpoint for live transcoding to MP4
# -----------------------------------------------------------------------------
@router.get("/api/media/transcode", tags=["media"])
def transcode_to_mp4(
    request: Request,
    path: str,
    height: int | None = 720,
    v_bitrate: str = "3000k",
    a_bitrate: str = "128k",
    start: float | None = None,
):
    """
    Live-transcode any source to H.264/AAC in a fragmented MP4 so browsers can play it.

    Query params:
      - path: absolute input file path (must be within ALLOWED_DIRECTORIES)
      - height: target video height (keep aspect). None to keep source size
      - v_bitrate: target video bitrate (e.g., "3000k")
      - a_bitrate: target audio bitrate (e.g., "128k")
      - start: optional start time in seconds

    Behavior:
      - Validates path & ffmpeg availability
      - Spawns ffmpeg and streams stdout to the client
      - Kills ffmpeg if the client disconnects
    """
    _ensure_ffmpeg_installed()

    raw_path = Path(path)
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not _is_under_allowed_roots(raw_path, ALLOWED_DIRECTORIES):
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")

    file_path = _safe_resolve(raw_path)
    cmd = _build_ffmpeg_cmd(file_path, height, v_bitrate, a_bitrate, start)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    async def gen():
        try:
            # Read stdout in small chunks and yield to client.
            # iterate_in_threadpool lets blocking .read() happen without blocking the event loop.
            while True:
                chunk = await iterate_in_threadpool(proc.stdout.read, STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                if await request.is_disconnected():
                    proc.kill()
                    break
                yield chunk
        finally:
            # Ensure process exits if we finish or client disconnects.
            try:
                proc.kill()
            except Exception:
                pass

    headers = {"Content-Type": "video/mp4"}
    return StreamingResponse(gen(), headers=headers)

# -----------------------------------------------------------------------------
# Endpoint for auto-play-URL
# -----------------------------------------------------------------------------
@router.get("/api/media/auto", tags=["media"])
def auto_play(request: Request, path: str):
    """
    One-stop play URL:
      - If the source is browser-native (MP4/H.264/AAC), stream directly.
      - Otherwise, live-transcode to a streamable MP4.
    """
    raw_path = Path(path)
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not _is_under_allowed_roots(raw_path, ALLOWED_DIRECTORIES):
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")

    file_path = _safe_resolve(raw_path)

    # Try probing; if probe fails, we consider it unsupported (and transcode).
    try:
        if _is_browser_native_playable(file_path):
            # Native path: redirect to your Range-enabled streamer
            return RedirectResponse(url=f"/api/media/stream?path={file_path}")
        else:
            # Non-native: redirect to transcode (defaults like 720p/3Mbps are fine to start)
            return RedirectResponse(url=f"/api/media/transcode?path={file_path}&height=720&v_bitrate=3000k&a_bitrate=128k")
    except HTTPException as e:
        if e.status_code == 415:
            # Explicitly unsupported: force transcode
            return RedirectResponse(url=f"/api/media/transcode?path={file_path}&height=720&v_bitrate=3000k&a_bitrate=128k")
        raise

# -----------------------------------------------------------------------------
# Helper function to safely resolve paths
# -----------------------------------------------------------------------------

def _safe_resolve(path: Path) -> Path:
    """
    Resolve the path safely (following symlinks) to an absolute canonical path.
    If resolution fails (broken symlink, permissions), raise a clean 404.
    """
    try:
        return path.resolve(strict=True)
    except Exception:
        # Don't leak filesystem details—just say "not found"
        raise HTTPException(status_code=404, detail="File not found")
    
# -----------------------------------------------------------------------------
# Helper function to check if a path is under allowed roots
# -----------------------------------------------------------------------------
def _is_under_allowed_roots(path: Path, roots: list[Path]) -> bool:
    """
    Return True if Path `path` is within one of the allowed root directories.

    Implementation detail:
    - We compare canonical absolute paths (resolved) to avoid symlink escapes.
    - Using Path.is_relative_to on 3.9+; for older versions use a try/except.
    """
    p_resolved = _safe_resolve(path)
    for root in roots:
        root_resolved = root.resolve()
        try:
            # Python 3.9+ has is_relative_to
            if p_resolved.is_relative_to(root_resolved):
                return True
        except AttributeError:
            # Fallback for older Python: emulate is_relative_to
            try:
                p_resolved.relative_to(root_resolved)
                return True
            except ValueError:
                pass
    return False

# -----------------------------------------------------------------------------
# Helper function to stream file ranges
# -----------------------------------------------------------------------------

def _iter_file_range(
    file_path: Path,                 # The absolute path to the file on disk
    start: int,                      # Byte offset to begin reading from (inclusive)
    end: int,                        # Byte offset to stop reading at (inclusive)
    chunk_size: int = STREAM_CHUNK_SIZE  # How many bytes to read and send at a time
) -> Iterator[bytes]:
    """
    Streams a file in chunks from `start` to `end` (inclusive).
    This generator yields small pieces of the file to StreamingResponse,
    allowing FastAPI to send data progressively instead of loading the whole file.

    Why this is important:
      - Keeps memory use constant no matter how big the file is.
      - Allows partial reads (important for HTTP Range requests and seeking).
      - Enables responsive video playback in browsers.
    """

    # Open the file in binary read mode ("rb" = read bytes).
    # Using a context manager ensures the file closes automatically
    # when we're done or if an exception occurs.
    with open(file_path, "rb") as f:

        # Move the file's read pointer to the requested start position.
        # This lets us begin reading partway through a file (for Range support).
        f.seek(start)

        # Calculate how many bytes we still need to read and send.
        # +1 because the 'end' position is inclusive.
        remaining = end - start + 1

        # Loop until we’ve sent all requested bytes to the client.
        while remaining > 0:

            # Decide how many bytes to read during this iteration.
            # For the final loop, 'remaining' may be smaller than 'chunk_size'.
            read_len = min(chunk_size, remaining)

            # Read the next block of data from disk.
            chunk = f.read(read_len)

            # If we didn't get any bytes back (EOF or read error), stop the loop.
            if not chunk:
                break

            # Subtract the number of bytes we just read from 'remaining'.
            remaining -= len(chunk)

            # Yield this chunk to the StreamingResponse.
            # FastAPI will send it to the client immediately,
            # then call this generator again for the next chunk.
            yield chunk

        # When the loop ends (either finished or early break),
        # the context manager automatically closes the file handle.

# -----------------------------------------------------------------------------
# Helper function to parse HTTP Range header
# -----------------------------------------------------------------------------

def _parse_range_header(range_header: str, file_size: int) -> Optional[tuple[int, int]]:
    """
    Parse a HTTP Range header of the form: "bytes=start-end"
    Return (start, end) as integers, or None if the header is invalid.
    Supports:
      - "bytes=START-"        (until EOF)
      - "bytes=-SUFFIX_LEN"   (last N bytes)
      - "bytes=START-END"     (explicit window)
    We do not support multiple ranges (comma-separated) for simplicity.
    """
    if not range_header:
        return None

    units, _, rng = range_header.partition("=")
    units = units.strip().lower()

    # Only byte ranges are supported
    if units != "bytes":
        return None

    # Some browsers might send multiple ranges (e.g., "bytes=0-1, 2-3"),
    # but we’ll keep MVP simple and only handle a single range.
    rng = rng.strip()
    if "," in rng:
        # Client asked for multiple ranges; you could implement multipart/byteranges,
        # but for MVP we’ll just reject it.
        return None

    start_str, _, end_str = rng.partition("-")
    start_str = start_str.strip()
    end_str = end_str.strip()

    # Case 1: "bytes=START-"
    if start_str and not end_str:
        try:
            start = int(start_str)
        except ValueError:
            return None
        if start >= file_size:
            return None
        return (start, file_size - 1)

    # Case 2: "bytes=-SUFFIX_LEN"  (last N bytes)
    if (not start_str) and end_str:
        try:
            suffix_len = int(end_str)
        except ValueError:
            return None
        if suffix_len <= 0:
            return None
        # If suffix_len > file_size, serve the whole file
        start = max(file_size - suffix_len, 0)
        end = file_size - 1
        return (start, end)

    # Case 3: "bytes=START-END"
    if start_str and end_str:
        try:
            start = int(start_str)
            end = int(end_str)
        except ValueError:
            return None
        if start > end or start >= file_size:
            return None
        # Clip end to file_size - 1
        end = min(end, file_size - 1)
        return (start, end)

    # Anything else is invalid
    return None

# -----------------------------------------------------------------------------
# Endpoint for file streaming (range)
# -----------------------------------------------------------------------------
@router.get("/api/media/stream", tags=["media"])
def stream_file(path: str, request: Request):
    """
    Stream a local file to the browser with proper HTTP Range support.

    Query params:
      - path: absolute path to a file previously discovered (must be under ALLOWED_DIRECTORIES)

    Behavior:
      - Validates path safety (canonical path must be within ALLOWED_DIRECTORIES).
      - If the client sends a Range header, respond 206 Partial Content and stream only that byte range.
      - Otherwise, respond 200 OK and stream the whole file.
      - Always stream in chunks to avoid loading large files into memory.
    """
    raw_path = Path(path)

    # 1) Validate existence and allowed roots (follows symlinks safely).
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not _is_under_allowed_roots(raw_path, ALLOWED_DIRECTORIES):
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")

    # Resolve to canonical path AFTER the checks to use for actual reading
    file_path = _safe_resolve(raw_path)

    # 2) Gather file info
    file_size = file_path.stat().st_size
    # Guess MIME type; fall back to octet-stream if unknown
    content_type, _ = mimetypes.guess_type(str(file_path))
    content_type = content_type or "application/octet-stream"

    # 3) Handle HTTP Range (seek) if provided
    range_header = request.headers.get("range") or request.headers.get("Range")
    byte_range = _parse_range_header(range_header, file_size) if range_header else None

    if byte_range:
        # Partial content
        start, end = byte_range
        # Build headers for a 206 response
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Content-Type": content_type,
            # Optional but sometimes useful:
            # "Cache-Control": "no-cache"
        }
        # Stream only the requested window
        return StreamingResponse(
            _iter_file_range(file_path, start, end),
            status_code=206,
            headers=headers,
        )

    # 4) No Range header → full file
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": content_type,
        # Optional cache policy:
        # "Cache-Control": "no-cache"
    }
    return StreamingResponse(_iter_file_range(file_path, 0, file_size - 1), headers=headers)