# backend/db.py
from __future__ import annotations

from pathlib import Path
import sqlite3
import os
from typing import List, Dict, Optional

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
DB_PATH = Path("./data/app.db")  # override with env if you want later
SQL_PATH = Path(__file__).parent / "sql" / "init.sql"

# Ensure ./data/ folder exists
os.makedirs(DB_PATH.parent, exist_ok=True)

# -----------------------------------------------------------------------------
# Connection helpers
# -----------------------------------------------------------------------------
def connect() -> sqlite3.Connection:
    """
    Create and return a connection to the SQLite database with sensible PRAGMAs.
    Each call returns a fresh connection; safe to use per request/operation.
    """
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row  # dict-like rows
    # Performance/safety knobs
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con

def init_db() -> None:
    """
    Ensure the schema exists by executing backend/sql/init.sql.
    Idempotent (uses IF NOT EXISTS in SQL).
    """
    if not SQL_PATH.exists():
        raise FileNotFoundError(f"Missing schema file: {SQL_PATH}")

    sql = SQL_PATH.read_text(encoding="utf-8")
    # Optional: strip accidental Python-style comments
    cleaned = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("#"))

    with connect() as con:
        con.executescript(cleaned)
        con.commit()

# -----------------------------------------------------------------------------
# Media operations (minimal)
# -----------------------------------------------------------------------------
def upsert_media_item(path: str, name: str) -> int:
    """
    Insert or update a media item by absolute path.
    Returns the media item's integer ID.
    """
    abs_path = str(Path(path))
    sql = """
    INSERT INTO media_items(path, name)
    VALUES(?, ?)
    ON CONFLICT(path) DO UPDATE SET
      name = excluded.name,
      updated_at = CURRENT_TIMESTAMP
    RETURNING id;
    """
    with connect() as con:
        row = con.execute(sql, (abs_path, name)).fetchone()
        con.commit()
        return int(row["id"])

def log_play(media_id: int) -> None:
    """
    Record a simple 'play' event for a given media_id.
    """
    with connect() as con:
        con.execute("INSERT INTO play_events(media_id) VALUES (?)", (media_id,))
        con.commit()

def get_by_path(path: str) -> Optional[Dict]:
    """
    Return media row by exact path, or None if not found.
    """
    abs_path = str(Path(path))
    with connect() as con:
        row = con.execute(
            "SELECT id, path, name, added_at, updated_at FROM media_items WHERE path = ?",
            (abs_path,),
        ).fetchone()
        return dict(row) if row else None

def get_recent(limit: int = 50) -> List[Dict]:
    """
    Return the most recently played/added items (simple home screen feed).
    Orders by last play time if available, otherwise by added_at.
    """
    sql = """
    SELECT
      m.id,
      m.name,
      m.path,
      m.added_at,
      m.updated_at,
      (
        SELECT MAX(played_at)
        FROM play_events p
        WHERE p.media_id = m.id
      ) AS last_played
    FROM media_items m
    ORDER BY COALESCE(last_played, m.added_at) DESC
    LIMIT ?
    """
    with connect() as con:
        rows = con.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in rows]

# -----------------------------------------------------------------------------
# NEW: Media lookups / listing to support router 'id=' flow
# -----------------------------------------------------------------------------
def get_media_by_id(media_id: int) -> Optional[Dict]:
    """
    Return media row by id, or None.
    """
    with connect() as con:
        row = con.execute(
            "SELECT id, path AS abs_path, name, added_at, updated_at FROM media_items WHERE id = ?",
            (media_id,),
        ).fetchone()
        return dict(row) if row else None

def list_media_items(limit: Optional[int] = None) -> List[Dict]:
    """
    Return media items; optionally limited.
    """
    base = "SELECT id, path AS abs_path, name, added_at, updated_at FROM media_items ORDER BY id DESC"
    params: tuple = ()
    if limit is not None:
        base += " LIMIT ?"
        params = (limit,)
    with connect() as con:
        rows = con.execute(base, params).fetchall()
        return [dict(r) for r in rows]

# -----------------------------------------------------------------------------
# NEW: Allowed directories management
# -----------------------------------------------------------------------------
def list_allowed_directories() -> List[str]:
    with connect() as con:
        rows = con.execute("SELECT path FROM allowed_directories ORDER BY added_at DESC").fetchall()
        return [r["path"] for r in rows]

def add_allowed_directory(path: str) -> None:
    """
    Add a directory to allowlist (absolute path). Idempotent.
    """
    abs_dir = str(Path(path).resolve())
    with connect() as con:
        con.execute(
            "INSERT INTO allowed_directories(path) VALUES (?) ON CONFLICT(path) DO NOTHING",
            (abs_dir,),
        )
        con.commit()

def remove_allowed_directory(path: str) -> None:
    """
    Remove a directory from allowlist (by exact stored path).
    """
    with connect() as con:
        con.execute("DELETE FROM allowed_directories WHERE path = ?", (path,))
        con.commit()

# -----------------------------------------------------------------------------
# CLI usage: initialize & quick verify
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Initializing database at: {DB_PATH}")
    init_db()
    with connect() as con:
        tables = [r["name"] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
    print("✅ Tables:", tables)
