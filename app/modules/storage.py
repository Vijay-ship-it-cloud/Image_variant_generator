"""
storage.py
----------
SQLite metadata + local "S3-style" file storage. Swap for PostgreSQL/S3 in
production -- this module is the single seam where that would happen.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
MASTERS_DIR = STORAGE_DIR / "masters"
VARIANTS_DIR = STORAGE_DIR / "variants"
DB_PATH = STORAGE_DIR / "app.db"

for d in (MASTERS_DIR, VARIANTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS masters (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                brand TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS variants (
                id TEXT PRIMARY KEY,
                master_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                variant_type TEXT NOT NULL,
                aspect_ratio TEXT,
                similarity_score REAL,
                passed_filter INTEGER NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (master_id) REFERENCES masters(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_logs (
                id TEXT PRIMARY KEY,
                master_id TEXT NOT NULL,
                variants_generated INTEGER,
                variants_filtered_out INTEGER,
                processing_time_sec REAL,
                created_at REAL NOT NULL
            )
            """
        )


def save_master(filename: str, image_bytes: bytes, brand: str | None = None) -> dict:
    master_id = str(uuid.uuid4())
    master_dir = MASTERS_DIR / master_id
    master_dir.mkdir(parents=True, exist_ok=True)
    filepath = master_dir / filename
    filepath.write_bytes(image_bytes)

    record = {
        "id": master_id,
        "filename": filename,
        "filepath": str(filepath),
        "brand": brand,
        "created_at": time.time(),
    }
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO masters (id, filename, filepath, brand, created_at) VALUES (?, ?, ?, ?, ?)",
            (record["id"], record["filename"], record["filepath"], record["brand"], record["created_at"]),
        )
    return record


def get_master(master_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM masters WHERE id = ?", (master_id,)).fetchone()
        return dict(row) if row else None


def list_masters() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM masters ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def save_variant(
    master_id: str,
    filename: str,
    image_bytes: bytes,
    variant_type: str,
    similarity_score: float,
    passed_filter: bool,
    aspect_ratio: str | None = None,
) -> dict:
    variant_id = str(uuid.uuid4())
    variant_dir = VARIANTS_DIR / master_id
    variant_dir.mkdir(parents=True, exist_ok=True)
    filepath = variant_dir / filename
    filepath.write_bytes(image_bytes)

    record = {
        "id": variant_id,
        "master_id": master_id,
        "filename": filename,
        "filepath": str(filepath),
        "variant_type": variant_type,
        "aspect_ratio": aspect_ratio,
        "similarity_score": similarity_score,
        "passed_filter": int(passed_filter),
        "created_at": time.time(),
    }
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO variants
              (id, master_id, filename, filepath, variant_type, aspect_ratio,
               similarity_score, passed_filter, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"], record["master_id"], record["filename"], record["filepath"],
                record["variant_type"], record["aspect_ratio"], record["similarity_score"],
                record["passed_filter"], record["created_at"],
            ),
        )
    return record


def list_variants_for_master(master_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM variants WHERE master_id = ? ORDER BY created_at ASC", (master_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def log_run(master_id: str, variants_generated: int, variants_filtered_out: int, processing_time_sec: float) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO run_logs (id, master_id, variants_generated, variants_filtered_out, processing_time_sec, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), master_id, variants_generated, variants_filtered_out, processing_time_sec, time.time()),
        )


def get_logs() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM run_logs ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def export_metadata_json(master_id: str) -> str:
    master = get_master(master_id)
    variants = list_variants_for_master(master_id)
    payload = {
        "master": master,
        "variants": [
            {
                "filename": v["filename"],
                "variant_type": v["variant_type"],
                "aspect_ratio": v["aspect_ratio"],
                "similarity_score": v["similarity_score"],
                "passed_filter": bool(v["passed_filter"]),
            }
            for v in variants
        ],
    }
    return json.dumps(payload, indent=2)