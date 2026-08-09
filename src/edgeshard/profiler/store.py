"""Profile store — SQLite persistence for profiling results.

This module provides storage for ProfileResult objects,
keyed by (model_name, model_revision, dtype, device_type, device_name).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from edgeshard.common.logging import get_logger
from edgeshard.profiler.profile_data import ProfileResult

logger = get_logger(__name__)

# Default database path
DEFAULT_DB_PATH = "profiles.db"

# SQL for creating the profiles table
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_revision TEXT NOT NULL DEFAULT '',
    dtype TEXT NOT NULL DEFAULT 'float16',
    device_type TEXT NOT NULL,
    device_name TEXT NOT NULL DEFAULT '',
    device_memory_mb INTEGER NOT NULL DEFAULT 0,
    layer_forward_ms REAL NOT NULL DEFAULT 0.0,
    kv_cache_per_token_mb REAL NOT NULL DEFAULT 0.0,
    prefill_tokens_per_sec REAL NOT NULL DEFAULT 0.0,
    decode_tokens_per_sec REAL NOT NULL DEFAULT 0.0,
    total_model_memory_mb REAL NOT NULL DEFAULT 0.0,
    num_layers_profiled INTEGER NOT NULL DEFAULT 0,
    num_runs INTEGER NOT NULL DEFAULT 10,
    timestamp REAL NOT NULL DEFAULT 0.0,
    UNIQUE(model_name, model_revision, dtype, device_type, device_name)
)
"""


class ProfileStore:
    """SQLite-based storage for profiling results.

    Profiles are uniquely identified by:
    (model_name, model_revision, dtype, device_type, device_name)

    Saving a profile with the same key will update the existing record.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        """Initialize ProfileStore.

        Args:
            db_path: Path to SQLite database file.
        """
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        try:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self._db_path)

    def save(self, result: ProfileResult) -> None:
        """Save a profiling result.

        If a profile with the same key exists, it will be updated.

        Args:
            result: ProfileResult to save.
        """
        conn = self._get_connection()
        try:
            # Use INSERT OR REPLACE to handle updates
            conn.execute(
                """
                INSERT OR REPLACE INTO profiles (
                    model_name, model_revision, dtype, device_type, device_name,
                    device_memory_mb, layer_forward_ms, kv_cache_per_token_mb,
                    prefill_tokens_per_sec, decode_tokens_per_sec,
                    total_model_memory_mb, num_layers_profiled, num_runs, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.model_name,
                    result.model_revision,
                    result.dtype,
                    result.device_type,
                    result.device_name,
                    result.device_memory_mb,
                    result.layer_forward_ms,
                    result.kv_cache_per_token_mb,
                    result.prefill_tokens_per_sec,
                    result.decode_tokens_per_sec,
                    result.total_model_memory_mb,
                    result.num_layers_profiled,
                    result.num_runs,
                    result.timestamp,
                ),
            )
            conn.commit()
            logger.info(f"Saved profile for {result.model_name} on {result.device_name}")
        finally:
            conn.close()

    def lookup(
        self,
        model_name: str,
        model_revision: str = "",
        dtype: str = "float16",
        device_type: str = "",
        device_name: str = "",
    ) -> ProfileResult | None:
        """Look up a profile by key.

        Args:
            model_name: Model name or path.
            model_revision: Model revision/hash.
            dtype: Weight dtype.
            device_type: Device type (e.g., "cuda:0").
            device_name: Device name (e.g., "RTX 4090").

        Returns:
            ProfileResult if found, None otherwise.
        """
        conn = self._get_connection()
        try:
            query = "SELECT * FROM profiles WHERE model_name = ? AND model_revision = ? AND dtype = ?"
            params: list[Any] = [model_name, model_revision, dtype]

            if device_type:
                query += " AND device_type = ?"
                params.append(device_type)
            if device_name:
                query += " AND device_name = ?"
                params.append(device_name)

            query += " ORDER BY timestamp DESC LIMIT 1"

            cursor = conn.execute(query, params)
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_result(row, cursor.description)
        finally:
            conn.close()

    def list_profiles(self) -> list[ProfileResult]:
        """List all stored profiles.

        Returns:
            List of ProfileResult objects, newest first.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM profiles ORDER BY timestamp DESC"
            )
            rows = cursor.fetchall()

            if not rows:
                return []

            return [self._row_to_result(row, cursor.description) for row in rows]
        finally:
            conn.close()

    def delete(self, model_name: str, device_name: str = "") -> bool:
        """Delete a profile.

        Args:
            model_name: Model name to delete.
            device_name: Optional device filter.

        Returns:
            True if a profile was deleted.
        """
        conn = self._get_connection()
        try:
            if device_name:
                cursor = conn.execute(
                    "DELETE FROM profiles WHERE model_name = ? AND device_name = ?",
                    (model_name, device_name),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM profiles WHERE model_name = ?",
                    (model_name,),
                )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted profile for {model_name}")
            return deleted
        finally:
            conn.close()

    def _row_to_result(
        self, row: tuple, description: list[tuple] | None
    ) -> ProfileResult:
        """Convert a database row to ProfileResult.

        Args:
            row: Database row tuple.
            description: Cursor description (column names).

        Returns:
            ProfileResult instance.
        """
        if description is None:
            # Fallback: assume standard column order
            return ProfileResult(
                model_name=row[1],
                model_revision=row[2],
                dtype=row[3],
                device_type=row[4],
                device_name=row[5],
                device_memory_mb=row[6],
                layer_forward_ms=row[7],
                kv_cache_per_token_mb=row[8],
                prefill_tokens_per_sec=row[9],
                decode_tokens_per_sec=row[10],
                total_model_memory_mb=row[11],
                num_layers_profiled=row[12],
                num_runs=row[13],
                timestamp=row[14],
            )

        # Use column names from description
        columns = [desc[0] for desc in description]
        data = dict(zip(columns, row))

        return ProfileResult(
            model_name=data["model_name"],
            model_revision=data["model_revision"],
            dtype=data["dtype"],
            device_type=data["device_type"],
            device_name=data["device_name"],
            device_memory_mb=data["device_memory_mb"],
            layer_forward_ms=data["layer_forward_ms"],
            kv_cache_per_token_mb=data["kv_cache_per_token_mb"],
            prefill_tokens_per_sec=data["prefill_tokens_per_sec"],
            decode_tokens_per_sec=data["decode_tokens_per_sec"],
            total_model_memory_mb=data["total_model_memory_mb"],
            num_layers_profiled=data["num_layers_profiled"],
            num_runs=data["num_runs"],
            timestamp=data["timestamp"],
        )
