-- backend/sql/init.sql
-- Idempotent schema for media items, play events, and allowed directories

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS media_items (
  id         INTEGER PRIMARY KEY,
  path       TEXT NOT NULL UNIQUE,           -- absolute path on disk
  name       TEXT NOT NULL,                  -- display name (usually stem)
  added_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_media_items_name ON media_items(name);

CREATE TABLE IF NOT EXISTS play_events (
  id         INTEGER PRIMARY KEY,
  media_id   INTEGER NOT NULL,
  played_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (media_id) REFERENCES media_items(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_play_events_media_id ON play_events(media_id);
CREATE INDEX IF NOT EXISTS idx_play_events_played_at ON play_events(played_at);

CREATE TABLE IF NOT EXISTS allowed_directories (
  id         INTEGER PRIMARY KEY,
  path       TEXT NOT NULL UNIQUE,           -- absolute directory path
  added_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Trigger to keep updated_at current on media_items updates
CREATE TRIGGER IF NOT EXISTS trg_media_items_updated
AFTER UPDATE ON media_items
FOR EACH ROW
BEGIN
  UPDATE media_items
  SET updated_at = CURRENT_TIMESTAMP
  WHERE id = NEW.id;
END;
