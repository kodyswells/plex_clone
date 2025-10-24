# backend/routers/media.py
# -----------------------------------------------------------------------------
# Media router:
#   - Startup scan of ALLOWED_DIRECTORIES for playable media
#   - List indexed paths            GET    /api/media/path/list
#   - Debug stat                    GET    /api/media/debug/stat?path=...
#   - Direct stream with Range      GET    /api/media/stream?path=...
#   - Live transcode (H.264/AAC)    GET    /api/media/transcode?path=...
#   - Auto-pick stream/transcode    GET    /api/media/auto?path=...
#
# Notes:
# - We never upload content; we only read local filesystem paths.
# - Path access is restricted to ALLOWED_DIRECTORIES.
# - Transcoding requires ffmpeg/ffprobe on PATH.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from urllib.parse import quote

router = APIRouter(tags=["media"])

# --- Configuration ------------------------------------------------------------

ALLOWED_DIRECTORIES: List[Path] = [
    Path("/home/kody/media"),  # adjust as needed; later make user-configurable via DB
]

# Normalize & keep only existing dirs (avoid surprises if a path disappears)
ALLOWED_DIRECTORIES = [p.resolve() for p in ALLOWED_DIRECTORIES if p.exists()]

MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".m4v", ".webm", ".mp3", ".aac", ".m4a", ".flac", ".wav", ".ogg", ".opus",
}

MAX_INDEXES = 50_000  # soft cap to avoid extreme scans
STREAM_CHUNK_SIZE = 64 * 1024  # 64 KiB

# In-memory index (cleared on restart)
file_list: List[Path] = []

# --- Helpers: filesystem & safety --------------------------------------------

def _contains_media_files(directory: Path) -> bool:
    try:
        for item in directory.iterdir():
            if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS and item.stat().st_size > 0:
                return True
    except Exception:
        pass
    return False


def _safe_resolve(path: Path) -> Path:
    """Resolve path safely to an absolute canonical path, raising a clean 404 on failure."""
    try:
        return path.resolve(strict=True)
    except Exception:
        raise HTTPException(status_code=404, detail="File not found")


def _is_under_allowed_roots(path: Path, roots: List[Path]) -> bool:
    """Return True if path is within one of the allowed root directories (after resolution)."""
    p_resolved = _safe_resolve(path)
    for root in roots:
        root_resolved = root.resolve()
        try:
            # Python 3.9+: Path.is_relative_to
            if p_resolved.is_relative_to(root_resolved):
                return True
        except AttributeError:
            try:
                p_resolved.relative_to(root_resolved)
                return True
            except ValueError:
                pass
    return False


def _iter_file_range(file_path: Path, start: int, end: int, chunk_size: int = STREAM_CHUNK_SIZE) -> Iterator[bytes]:
    """Yield bytes from file_path in [start, end] inclusive, chunked for streaming."""
    with open(file_path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            read_len = min(chunk_size, remaining)
            chunk = f.read(read_len)
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _parse_range_header(range_header: str | None, file_size: int) -> Optional[Tuple[int, int]]:
    """Parse 'Range: bytes=...' into (start, end), or None if invalid/unsupported."""
    if not range_header:
        return None

    units, _, rng = range_header.partition("=")
    if units.strip().lower() != "bytes":
        return None

    rng = rng.strip()
    if "," in rng:
        # Multiple ranges not supported in MVP
        return None

    start_str, _, end_str = rng.partition("-")
    start_str, end_str = start_str.strip(), end_str.strip()

    if start_str and not end_str:
        try:
            start = int(start_str)
        except ValueError:
            return None
        if start >= file_size:
            return None
        return (start, file_size - 1)

    if (not start_str) and end_str:
        try:
            suffix_len = int(end_str)
        except ValueError:
            return None
        if suffix_len <= 0:
            return None
        start = max(file_size - suffix_len, 0)
        return (start, file_size - 1)

    if start_str and end_str:
        try:
            start = int(start_str)
            end = int(end_str)
        except ValueError:
            return None
        if start > end or start >= file_size:
            return None
        return (start, min(end, file_size - 1))

    return None

# --- ffmpeg/ffprobe helpers ---------------------------------------------------

def _ensure_ffmpeg_installed() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise HTTPException(status_code=500, detail="ffmpeg/ffprobe not found on PATH.")


def _ffprobe_streams(path: Path) -> dict:
    """
    Return {'container','video_codec','audio_codec'} for given media file using ffprobe.
    Raises 415 if probe fails.
    """
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
            stderr=subprocess.STDOUT,
        )
        data = json.loads(out.decode("utf-8", errors="ignore"))
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=415, detail=f"Unable to probe media: {e.output.decode(errors='ignore')[:200]}")
    except Exception:
        raise HTTPException(status_code=415, detail="Unable to probe media.")

    container = (data.get("format") or {}).get("format_name", "")
    vcodec = None
    acodec = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and vcodec is None:
            vcodec = s.get("codec_name")
        if s.get("codec_type") == "audio" and acodec is None:
            acodec = s.get("codec_name")
    return {"container": container, "video_codec": vcodec, "audio_codec": acodec}


def _is_browser_native_playable(path: Path) -> bool:
    """Conservative rule: MP4(-like) container + H.264 video + AAC audio."""
    info = _ffprobe_streams(path)
    container = (info["container"] or "").lower()
    mp4_like = any(name in container for name in ("mp4", "mov", "m4a", "3gp", "3g2", "mj2"))
    v_ok = (info["video_codec"] or "").lower() == "h264"
    a_ok = (info["audio_codec"] or "").lower() == "aac"
    return mp4_like and v_ok and a_ok


def _nvenc_runtime_ok() -> bool:
    """True if NVIDIA runtime looks usable (nvidia-smi reports at least one GPU)."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, timeout=1)
        return b"GPU" in out
    except Exception:
        return False


def _pick_encoder(prefer: Optional[str] = None) -> str:
    """Pick best H.264 encoder (NVENC if available/usable; else libx264)."""
    if prefer == "nvenc" and _nvenc_runtime_ok():
        return "h264_nvenc"
    try:
        encoders_out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT).decode(
            "utf-8", errors="ignore"
        ).lower()
        if "h264_nvenc" in encoders_out and _nvenc_runtime_ok():
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


def _build_ffmpeg_cmd(
    path: Path,
    height: int | None,
    v_bitrate: str,
    a_bitrate: str,
    start: float | None,
    prefer_encoder: str | None = None,
) -> list[str]:
    encoder = _pick_encoder(prefer_encoder)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start and start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]

    filters = ["format=yuv420p"]
    if height:
        filters.insert(0, f"scale=-2:{height}")
    if filters:
        cmd += ["-vf", ",".join(filters)]

    if encoder == "h264_nvenc":
        cmd += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            "23",
            "-b:v",
            "0",
            "-g",
            "48",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
        ]
    else:
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "film",
            "-b:v",
            v_bitrate,
            "-maxrate",
            v_bitrate,
            "-bufsize",
            str(int(int(v_bitrate.rstrip("k")) * 2)) + "k",
            "-g",
            "48",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
        ]

    cmd += [
        "-c:a",
        "aac",
        "-b:a",
        a_bitrate,
        "-movflags",
        "+faststart+frag_keyframe+empty_moov+default_base_moof",
        "-f",
        "mp4",
        "pipe:1",
    ]
    return cmd

# --- Startup scan -------------------------------------------------------------

_scan_ran = False

@router.on_event("startup")
def startup_event():
    global _scan_ran
    if _scan_ran:
        return  # already ran
    _scan_ran = True

    file_list.clear()  # avoid accumulating duplicates
    total_indexed = 0
    for directory in ALLOWED_DIRECTORIES:
        if not directory.exists() or not directory.is_dir():
            print(f"[media] Skipping missing dir: {directory}")
            continue
        dir_count = 0
        for item in directory.rglob("*"):
            if len(file_list) >= MAX_INDEXES:
                break
            if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS:
                try:
                    if item.stat().st_size > 0 and _is_under_allowed_roots(item, ALLOWED_DIRECTORIES):
                        file_list.append(item.resolve())
                        dir_count += 1
                except Exception:
                    continue
        total_indexed += dir_count
        print(f"[media] Indexed {dir_count} media files from {directory}")
    print(f"[media] Startup total indexed: {total_indexed}")

# --- Endpoints ----------------------------------------------------------------

@router.get("/media/path/list")
def get_path_list():
    """Return the list of indexed media file paths."""
    return [str(path) for path in file_list]


@router.get("/media/debug/stat")
def debug_stat(path: str):
    p = Path(path)
    info = {"path": path}
    info["exists"] = p.exists()
    info["is_file"] = p.is_file()
    try:
        info["resolved"] = str(p.resolve(strict=True))
    except Exception as e:
        info["resolved_error"] = str(e)
    try:
        info["allowed"] = _is_under_allowed_roots(p, ALLOWED_DIRECTORIES)
    except Exception as e:
        info["allowed_error"] = str(e)
    return info


@router.get("/media/stream")
def stream_file(path: str, request: Request):
    """
    Stream a local file with HTTP Range support.
    """
    raw_path = Path(path)
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not _is_under_allowed_roots(raw_path, ALLOWED_DIRECTORIES):
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")

    file_path = _safe_resolve(raw_path)
    file_size = file_path.stat().st_size
    content_type, _ = mimetypes.guess_type(str(file_path))
    content_type = content_type or "application/octet-stream"

    range_header = request.headers.get("range") or request.headers.get("Range")
    byte_range = _parse_range_header(range_header, file_size) if range_header else None

    if byte_range:
        start, end = byte_range
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Content-Type": content_type,
        }
        return StreamingResponse(_iter_file_range(file_path, start, end), status_code=206, headers=headers)

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": content_type,
    }
    return StreamingResponse(_iter_file_range(file_path, 0, file_size - 1), headers=headers)


@router.get("/media/transcode")
def transcode_to_mp4(
    request: Request,
    path: str,
    height: int | None = 720,
    v_bitrate: str = "3000k",
    a_bitrate: str = "128k",
    start: float | None = None,
    prefer_encoder: str | None = None,  # e.g., "nvenc"
):
    """
    Live-transcode any source video to H.264/AAC MP4 and stream it.
    IMPORTANT: stdout/stderr are kept in BINARY. Do NOT set text=True on Popen.
    """
    _ensure_ffmpeg_installed()

    raw_path = Path(path)
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not _is_under_allowed_roots(raw_path, ALLOWED_DIRECTORIES):
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")

    file_path = _safe_resolve(raw_path)
    cmd = _build_ffmpeg_cmd(file_path, height, v_bitrate, a_bitrate, start, prefer_encoder)

    # NOTE: text=False (default). stdout is raw MP4 bytes, do NOT decode.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # keep logs; read them in a side thread
        bufsize=0,
    )

    # Drain stderr in a background thread so its pipe never fills up
    def _drain_stderr():
        try:
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                # Decode for console printing but ignore errors
                try:
                    sys.stderr.write(f"[ffmpeg] {line.decode('utf-8', errors='ignore')}")
                except Exception:
                    pass
        except Exception:
            pass

    threading.Thread(target=_drain_stderr, daemon=True).start()

    async def gen():
        try:
            while True:
                # Read raw bytes in a worker thread to avoid blocking the loop
                chunk = await run_in_threadpool(proc.stdout.read, STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                # If client disconnected, stop ffmpeg
                if await request.is_disconnected():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break
                yield chunk
        finally:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        await run_in_threadpool(proc.wait, 2)
                    except Exception:
                        proc.kill()
            except Exception:
                pass
    encoder = _pick_encoder(prefer_encoder)
    print(f"[media] Starting transcode: {file_path} -> H.264/AAC (encoder={encoder}, height={height}, v_bitrate={v_bitrate}, a_bitrate={a_bitrate}, start={start})")
    # No Content-Length/Range for live output
    return StreamingResponse(gen(), media_type="video/mp4", headers={"Cache-Control": "no-store", "X-Transcoder-Encoder": encoder})


@router.get("/media/auto")
def auto_play(request: Request, path: str):
    """
    If the source is browser-native (MP4/H.264/AAC), redirect to /media/stream.
    Otherwise, redirect to /media/transcode with sensible defaults.
    """
    raw_path = Path(path)
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not _is_under_allowed_roots(raw_path, ALLOWED_DIRECTORIES):
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")

    file_path = _safe_resolve(raw_path)
    qp = quote(str(file_path))
    try:
        if _is_browser_native_playable(file_path):
            return RedirectResponse(url=f"/api/media/stream?path={qp}")
        else:
            return RedirectResponse(url=f"/api/media/transcode?path={qp}&height=720&v_bitrate=3000k&a_bitrate=128k")
    except HTTPException as e:
        # If probing fails (415), just transcode
        if e.status_code == 415:
            return RedirectResponse(url=f"/api/media/transcode?path={qp}&height=720&v_bitrate=3000k&a_bitrate=128k")
        raise
