# backend/routers/media.py
# -----------------------------------------------------------------------------
# Media router (DB-backed):
#   - Startup scan of SQLite "allowed directories" for playable media
#     (also upserts media items to DB)
#   - Manage allowed dirs:
#       GET    /media/allowed-dirs
#       POST   /media/allowed-dirs?path=/abs/dir
#       DELETE /media/allowed-dirs?path=/abs/dir
#   - List indexed paths (in-memory, for quick debug)
#       GET    /media/path/list
#   - Optional: list media items from DB
#       GET    /media/item/list
#   - Debug stat
#       GET    /media/debug/stat?path=...
#   - Direct stream with Range      GET    /media/stream?id=... | path=...
#   - Live transcode (H.264/AAC)    GET    /media/transcode?id=... | path=...
#   - Auto-pick stream/transcode    GET    /media/auto?id=... | path=...
#
# Notes:
# - We never upload content; we only read local filesystem paths.
# - Path access is restricted to the SQLite-backed allowed directories.
# - Transcoding requires ffmpeg/ffprobe on PATH.
# - Persists minimal info to SQLite and logs play events.
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
from urllib.parse import urlencode

# ✅ DB hooks (adjust names here if your backend.db differs)
from backend.db import (
    list_allowed_directories,
    add_allowed_directory,
    remove_allowed_directory,
    upsert_media_item,
    log_play,
    get_media_by_id,
    list_media_items,  # optional, used by /media/item/list
)

router = APIRouter(tags=["media"])
LOCK = threading.Lock()

# Transcoding settings (runtime-configurable)
transcode_settings = {
    "height": 720,
    "v_bitrate": "3000k",
    "a_bitrate": "128k",
    "prefer_encoder": None,  # e.g., "nvenc"
}

# --- Configuration ------------------------------------------------------------

hardware_acceleration = False  # Disable hardware acceleration by default

MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".m4v", ".webm", ".mp3", ".aac", ".m4a", ".flac", ".wav", ".ogg", ".opus",
}

MAX_INDEXES = 50_000  # soft cap to avoid extreme scans
STREAM_CHUNK_SIZE = 64 * 1024  # 64 KiB

# In-memory index (cleared on restart)
file_list: List[Path] = []

# Server-side directory browser roots (where browsing can begin)
BROWSER_ROOTS = [
    Path("/home"),
    Path("/mnt"),
    Path("/media"),
    Path("/srv"),
    Path("/storage"),
]


# --- Helpers: filesystem & safety --------------------------------------------

def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except Exception:
        raise HTTPException(status_code=404, detail="File not found")
    
def _norm_dir(p: Path) -> Path:
    p = p.expanduser()
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="Directory not found.")
    return p.resolve()

def _list_directory(path: Path) -> list[dict]:
    """Return subdirectories of path (no files) with basic metadata."""
    out = []
    try:
        for entry in sorted(path.iterdir(), key=lambda x: x.name.lower()):
            # Show only directories; skip unreadable or broken links gracefully
            try:
                if entry.is_dir():
                    out.append({
                        "name": entry.name,
                        "path": str(entry.resolve()),
                        "type": "dir",
                    })
            except Exception:
                continue
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied.")
    return out


def _is_browser_native_playable(path: Path) -> bool:
    """
    Return True if the file is directly playable by most browsers
    (MP4 container + H.264 video + AAC audio).
    """
    try:
        info = _ffprobe_streams(path)
    except HTTPException:
        return False

    container = (info["container"] or "").lower()
    mp4_like = any(x in container for x in ("mp4", "mov", "m4a", "3gp", "3g2", "mj2"))
    v_ok = (info["video_codec"] or "").lower() == "h264"
    a_ok = (info["audio_codec"] or "").lower() == "aac"
    return mp4_like and v_ok and a_ok



def _db_allowed_roots() -> List[Path]:
    """Fetch allowed directories from DB, normalize & filter existing ones."""
    roots: List[Path] = []
    try:
        for p in list_allowed_directories() or []:
            pp = Path(p).expanduser()
            if pp.exists() and pp.is_dir():
                roots.append(pp.resolve())
    except Exception as e:
        print(f"[media] list_allowed_directories() failed: {e}")
    return roots


def _is_under_allowed_roots(path: Path, roots: List[Path] | None = None) -> bool:
    """Return True if path is within one of the allowed root directories (after resolution)."""
    p_resolved = _safe_resolve(path)
    roots = roots if roots is not None else _db_allowed_roots()
    for root in roots:
        root_resolved = root.resolve()
        try:
            if p_resolved.is_relative_to(root_resolved):  # py3.9+
                return True
        except AttributeError:
            try:
                p_resolved.relative_to(root_resolved)
                return True
            except ValueError:
                pass
    return False


def _iter_file_range(file_path: Path, start: int, end: int, chunk_size: int = STREAM_CHUNK_SIZE) -> Iterator[bytes]:
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
    if not range_header:
        return None
    units, _, rng = range_header.partition("=")
    if units.strip().lower() != "bytes":
        return None
    rng = rng.strip()
    if "," in rng:
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


def _nvenc_runtime_ok() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, timeout=1)
        return b"GPU" in out
    except Exception:
        return False


def _pick_encoder(prefer: Optional[str] = None) -> str:
    if prefer == "nvenc" and _nvenc_runtime_ok():
        return "h264_nvenc"
    try:
        encoders_out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT).decode(
            "utf-8", errors="ignore"
        ).lower()
        if "h264_nvenc" in encoders_out and _nvenc_runtime_ok() and hardware_acceleration is True:
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
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", "23",
            "-b:v", "0",
            "-g", "48",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "film",
            "-b:v", v_bitrate,
            "-maxrate", v_bitrate,
            "-bufsize", str(int(int(v_bitrate.rstrip("k")) * 2)) + "k",
            "-g", "48",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
        ]

    cmd += [
        "-c:a", "aac",
        "-b:a", a_bitrate,
        "-movflags", "+faststart+frag_keyframe+empty_moov+default_base_moof",
        "-f", "mp4",
        "pipe:1",
    ]
    return cmd

# --- DB helper ---------------------------------------------------------------

def _record_media_event(abs_path: Path) -> None:
    try:
        media_id = upsert_media_item(str(abs_path), abs_path.stem)
        log_play(media_id)
    except Exception as e:
        print(f"[media] DB record/log_play failed for {abs_path}: {e}")

# --- Startup scan -------------------------------------------------------------

_scan_ran = False

@router.on_event("startup")
def startup_event():
    """
    Load allowed directories from DB, scan for media (bounded by MAX_INDEXES),
    upsert to DB, and refresh in-memory file_list (debug/compat).
    """
    global _scan_ran
    if _scan_ran:
        return
    _scan_ran = True

    roots = _db_allowed_roots()
    if not roots:
        print("[media] No allowed directories in DB yet. Add one via POST /api/media/allowed-dirs?path=/abs/dir")
        return

    file_list.clear()
    total_indexed = 0

    for directory in roots:
        if not directory.exists() or not directory.is_dir():
            print(f"[media] Skipping missing dir: {directory}")
            continue

        dir_count = 0
        for item in directory.rglob("*"):
            if len(file_list) >= MAX_INDEXES:
                break
            if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS:
                try:
                    if item.stat().st_size > 0 and _is_under_allowed_roots(item, roots):
                        resolved = item.resolve()
                        file_list.append(resolved)
                        dir_count += 1
                        try:
                            upsert_media_item(str(resolved), resolved.stem)
                        except Exception as e:
                            print(f"[media] upsert_media_item failed for {resolved}: {e}")
                except Exception:
                    continue
        total_indexed += dir_count
        print(f"[media] Indexed {dir_count} media files from {directory}")
    print(f"[media] Startup total indexed: {total_indexed}")

# --- Allowed directories management ------------------------------------------

@router.get("/media/allowed-dirs")
def get_allowed_dirs():
    """List allowed root directories from SQLite."""
    try:
        return list_allowed_directories() or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list allowed dirs: {e}")


@router.post("/media/allowed-dirs")
def add_allowed_dir(path: str):
    """
    Add an absolute directory to the allowlist and trigger a light scan of that dir.
    """
    p = Path(path).expanduser()
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="Directory does not exist.")
    try:
        add_allowed_directory(str(p.resolve()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add allowed dir: {e}")

    # Light scan of just this dir
    added = 0
    for item in p.rglob("*"):
        if added >= MAX_INDEXES:
            break
        if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS:
            try:
                if item.stat().st_size > 0 and _is_under_allowed_roots(item):
                    resolved = item.resolve()
                    try:
                        upsert_media_item(str(resolved), resolved.stem)
                        added += 1
                    except Exception as e:
                        print(f"[media] upsert failed for {resolved}: {e}")
                    # keep the in-mem list as a convenience
                    file_list.append(resolved)
            except Exception:
                continue

    return {"status": "ok", "added_media": added, "dir": str(p.resolve())}


@router.delete("/media/allowed-dirs")
def delete_allowed_dir(path: str):
    """
    Remove a directory from the allowlist. (Does not delete media rows.)
    """
    try:
        remove_allowed_directory(path)
        return {"status": "ok", "removed": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove allowed dir: {e}")

# --- Endpoints ----------------------------------------------------------------

@router.get("/media/path/list")
def get_path_list():
    """Return the in-memory list of indexed media file paths (debug/compat)."""
    return [str(path) for path in file_list]


@router.get("/media/item/list")
def get_media_item_list(limit: int | None = 500):
    """
    Convenience endpoint that returns media items from the DB
    (id, abs_path, name, etc.). 'limit' is optional.
    """
    try:
        items = list_media_items(limit=limit)
        return items
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch media items: {e}")


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
        info["allowed"] = _is_under_allowed_roots(p)
    except Exception as e:
        info["allowed_error"] = str(e)
    return info


def _resolve_path_from_id_or_path(id: int | None, path: str | None) -> Path:
    """
    Prefer DB id if provided; else fall back to direct path (kept for compatibility).
    """
    if id is not None:
        row = get_media_by_id(int(id))
        if not row or not row.get("abs_path"):
            raise HTTPException(status_code=404, detail="Media id not found")
        resolved = Path(row["abs_path"])
    else:
        if not path:
            raise HTTPException(status_code=400, detail="Provide 'id' or 'path'")
        resolved = Path(path)

    if not resolved.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not _is_under_allowed_roots(resolved):
        raise HTTPException(status_code=403, detail="Access to this path is not allowed")
    return _safe_resolve(resolved)


@router.get("/media/stream")
def stream_file(request: Request, id: int | None = None, path: str | None = None):
    """
    Stream a local file with HTTP Range support.
    Prefer 'id' (SQLite) over 'path'.
    """
    file_path = _resolve_path_from_id_or_path(id, path)

    # ensure DB has this item + log play
    _record_media_event(file_path)

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
    id: int | None = None,
    path: str | None = None,
    height: int = transcode_settings["height"],
    v_bitrate: str = transcode_settings["v_bitrate"],
    a_bitrate: str = transcode_settings["a_bitrate"],
    start: float | None = None,
    prefer_encoder: str | None = None,
):
    """
    Live-transcode any source video to H.264/AAC MP4 and stream it.
    Supports 'id' (preferred) or 'path'.
    """
    _ensure_ffmpeg_installed()
    file_path = _resolve_path_from_id_or_path(id, path)

    # ensure DB has this item + log play
    _record_media_event(file_path)

    cmd = _build_ffmpeg_cmd(file_path, height, v_bitrate, a_bitrate, start, prefer_encoder)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # keep logs; read them in a side thread
        bufsize=0,
    )

    def _drain_stderr():
        try:
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
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
                chunk = await run_in_threadpool(proc.stdout.read, STREAM_CHUNK_SIZE)
                if not chunk:
                    break
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
    return StreamingResponse(gen(), media_type="video/mp4", headers={"Cache-Control": "no-store", "X-Transcoder-Encoder": encoder})


@router.get("/media/auto")
def auto_play(request: Request, id: int | None = None, path: str | None = None):
    """
    If the source is browser-native (MP4/H.264/AAC), redirect to /media/stream.
    Otherwise, redirect to /media/transcode with current transcoding settings.
    Accepts 'id' (preferred) or 'path'.
    """
    file_path = _resolve_path_from_id_or_path(id, path)
    try:
        if _is_browser_native_playable(file_path):
            qp = {"path": str(file_path)}  # keep path for stream (no re-lookup)
            return RedirectResponse(url=f"/api/media/stream?{urlencode(qp)}")
        else:
            with LOCK:
                current_settings = transcode_settings.copy()
            qp = {
                "path": str(file_path),
                "height": current_settings["height"],
                "v_bitrate": current_settings["v_bitrate"],
                "a_bitrate": current_settings["a_bitrate"],
            }
            if current_settings.get("prefer_encoder"):
                qp["prefer_encoder"] = current_settings["prefer_encoder"]
            return RedirectResponse(url=f"/api/media/transcode?{urlencode(qp)}")
    except HTTPException as e:
        if e.status_code == 415:
            with LOCK:
                current_settings = transcode_settings.copy()
            qp = {
                "path": str(file_path),
                "height": current_settings["height"],
                "v_bitrate": current_settings["v_bitrate"],
                "a_bitrate": current_settings["a_bitrate"],
            }
            if current_settings.get("prefer_encoder"):
                qp["prefer_encoder"] = current_settings["prefer_encoder"]
            return RedirectResponse(url=f"/api/media/transcode?{urlencode(qp)}")
        raise

# --- Transcode config ---------------------------------------------------------

@router.post("/media/transcode/configure")
def configure_transcode_settings(height: int):
    with LOCK:
        if height not in (360, 480, 720, 1080):
            raise HTTPException(status_code=400, detail="Unsupported height. Supported heights are 360, 480, 720, and 1080")

        transcode_settings["height"] = height
        if height == 360:
            transcode_settings["v_bitrate"] = "1000k"
            transcode_settings["a_bitrate"] = "96k"
        elif height == 480:
            transcode_settings["v_bitrate"] = "1500k"
            transcode_settings["a_bitrate"] = "112k"
        elif height == 720:
            transcode_settings["v_bitrate"] = "3000k"
            transcode_settings["a_bitrate"] = "128k"
        elif height == 1080:
            transcode_settings["v_bitrate"] = "5000k"
            transcode_settings["a_bitrate"] = "192k"

    return {"status": "success", "transcode_settings": transcode_settings}


@router.post("/media/transcode/hardware_acceleration")
def set_hardware_acceleration(enabled: bool):
    with LOCK:
        global hardware_acceleration
        hardware_acceleration = enabled
        return {"status": "success", "hardware_acceleration": hardware_acceleration}


@router.get("/media/transcode/settings")
def get_transcode_settings():
    with LOCK:
        return transcode_settings


@router.get("/media/fs/roots")
def fs_roots():
    """Starting points for the server-side browser."""
    roots = []
    for p in BROWSER_ROOTS:
        try:
            if p.exists() and p.is_dir():
                roots.append(str(p.resolve()))
        except Exception:
            continue
    # Also expose currently allowed dirs as convenient shortcuts
    try:
        roots.extend(list_allowed_directories() or [])
    except Exception:
        pass
    # De-dup while preserving order
    seen = set(); uniq = []
    for r in roots:
        if r not in seen:
            uniq.append(r); seen.add(r)
    return uniq

@router.get("/media/fs/list")
def fs_list(path: str):
    """
    List subdirectories of 'path' on the server.
    Read-only. Returns breadcrumbs and child directories.
    """
    cur = _norm_dir(Path(path))
    # Build breadcrumbs
    crumbs = []
    parts = cur.parts
    for i in range(1, len(parts) + 1):
        crumbs.append(str(Path(*parts[:i])))
    return {
        "path": str(cur),
        "parent": str(cur.parent) if cur != cur.parent else None,
        "breadcrumbs": crumbs,
        "dirs": _list_directory(cur),
    }
