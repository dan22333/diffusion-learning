from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import re
import sqlite3
import threading
from itertools import islice
from pathlib import Path
from typing import Any

from .model import PRIVATE_PARTS, viewer_kind


MAX_PREVIEW_BYTES = 4 * 1024 * 1024
MAX_PAGE_SIZE = 500


class PathPolicy:
    def __init__(self, root: Path, notebook: Path, experiments_glob: str) -> None:
        self.root = root.resolve()
        self.notebook = notebook.resolve()
        self.experiments_glob = experiments_glob

    def experiment_roots(self) -> list[Path]:
        return [path.resolve() for path in self.root.glob(self.experiments_glob) if path.is_dir()]

    def resolve(self, raw: str, *, must_exist: bool = True) -> Path:
        candidate = (self.root / raw).resolve()
        if PRIVATE_PARTS.intersection(candidate.parts):
            raise PermissionError("private extraction content is unavailable")
        allowed = candidate == self.notebook or self.notebook in candidate.parents or any(
            candidate == experiment or experiment in candidate.parents for experiment in self.experiment_roots()
        )
        if not allowed:
            raise PermissionError("path is outside the notebook and experiment roots")
        if must_exist and not candidate.is_file():
            raise FileNotFoundError(raw)
        return candidate

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()


class TableCache:
    def __init__(self, database: Path, policy: PathPolicy) -> None:
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.lock = threading.RLock()
        self.importing: set[str] = set()
        self.connection = sqlite3.connect(database, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        with self.lock:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS artifact_tables (path TEXT PRIMARY KEY, mtime_ns INTEGER, size INTEGER, table_name TEXT, status TEXT, headers TEXT, row_count INTEGER, error TEXT)"
            )
            self.connection.commit()

    @staticmethod
    def _delimiter(path: Path) -> str:
        return "\t" if path.suffix.lower() == ".tsv" else ","

    @staticmethod
    def _table_name(relative: str) -> str:
        return "artifact_" + hashlib.sha256(relative.encode()).hexdigest()[:20]

    def _metadata(self, relative: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute("SELECT * FROM artifact_tables WHERE path = ?", (relative,)).fetchone()
        return dict(row) if row else None

    def _fresh(self, path: Path, metadata: dict[str, Any] | None) -> bool:
        if not metadata or metadata.get("status") != "ready":
            return False
        stat = path.stat()
        return metadata.get("mtime_ns") == stat.st_mtime_ns and metadata.get("size") == stat.st_size

    def ensure_async(self, path: Path) -> dict[str, Any]:
        relative = self.policy.relative(path)
        metadata = self._metadata(relative)
        if self._fresh(path, metadata):
            return metadata or {}
        with self.lock:
            if relative not in self.importing:
                self.importing.add(relative)
                thread = threading.Thread(target=self._import, args=(path,), daemon=True, name=f"artifact-index-{path.name}")
                thread.start()
        return metadata or {"path": relative, "status": "indexing", "headers": "[]", "row_count": None}

    def _import(self, path: Path) -> None:
        relative = self.policy.relative(path)
        table_name = self._table_name(relative)
        stat = path.stat()
        headers: list[str] = []
        row_count = 0
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.reader(handle, delimiter=self._delimiter(path))
                headers = next(reader, [])
                columns = [f"c{index}" for index in range(len(headers))]
                with self.lock:
                    self.connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                    schema = ", ".join(f'"{column}" TEXT' for column in columns)
                    self.connection.execute(f'CREATE TABLE "{table_name}" (row_number INTEGER PRIMARY KEY, {schema})')
                    self.connection.execute(
                        "INSERT OR REPLACE INTO artifact_tables VALUES (?,?,?,?,?,?,?,?)",
                        (relative, stat.st_mtime_ns, stat.st_size, table_name, "indexing", json.dumps(headers), None, ""),
                    )
                    self.connection.commit()
                placeholders = ",".join("?" for _ in range(len(columns) + 1))
                batch: list[tuple[Any, ...]] = []
                for row_count, row in enumerate(reader, start=1):
                    row = (row + [""] * len(columns))[: len(columns)]
                    batch.append((row_count, *row))
                    if len(batch) >= 1000:
                        with self.lock:
                            self.connection.executemany(f'INSERT INTO "{table_name}" VALUES ({placeholders})', batch)
                            self.connection.commit()
                        batch.clear()
                if batch:
                    with self.lock:
                        self.connection.executemany(f'INSERT INTO "{table_name}" VALUES ({placeholders})', batch)
                        self.connection.commit()
            with self.lock:
                self.connection.execute(
                    "UPDATE artifact_tables SET status='ready', headers=?, row_count=?, error='' WHERE path=?",
                    (json.dumps(headers), row_count, relative),
                )
                self.connection.commit()
        except Exception as error:
            with self.lock:
                self.connection.execute(
                    "INSERT OR REPLACE INTO artifact_tables VALUES (?,?,?,?,?,?,?,?)",
                    (relative, stat.st_mtime_ns, stat.st_size, table_name, "error", json.dumps(headers), row_count, str(error)),
                )
                self.connection.commit()
        finally:
            with self.lock:
                self.importing.discard(relative)

    def _direct_page(self, path: Path, page_size: int) -> tuple[list[str], list[list[str]]]:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=self._delimiter(path))
            headers = next(reader, [])
            rows = list(islice(reader, page_size))
        return headers, rows

    def page(
        self,
        raw_path: str,
        page: int = 0,
        page_size: int = 100,
        sort_column: int | None = None,
        direction: str = "asc",
        filter_text: str = "",
    ) -> dict[str, Any]:
        path = self.policy.resolve(raw_path)
        if viewer_kind(path) != "table":
            raise ValueError("artifact is not a table")
        page, page_size = max(0, page), max(1, min(MAX_PAGE_SIZE, page_size))
        metadata = self.ensure_async(path)
        if not self._fresh(path, metadata):
            headers, rows = self._direct_page(path, page_size)
            return {"headers": headers, "rows": rows, "page": 0, "page_size": page_size, "row_count": None, "indexing": True}
        headers = json.loads(metadata["headers"])
        table_name = metadata["table_name"]
        conditions = ""
        params: list[Any] = []
        if filter_text:
            conditions = " WHERE " + " OR ".join(f'"c{index}" LIKE ?' for index in range(len(headers)))
            params.extend([f"%{filter_text}%"] * len(headers))
        order = "row_number ASC"
        if sort_column is not None and 0 <= sort_column < len(headers):
            order = f'"c{sort_column}" {"DESC" if direction.lower() == "desc" else "ASC"}'
        params.extend([page_size, page * page_size])
        with self.lock:
            rows = self.connection.execute(
                f'SELECT * FROM "{table_name}"{conditions} ORDER BY {order} LIMIT ? OFFSET ?', params
            ).fetchall()
        values = [[row[f"c{index}"] for index in range(len(headers))] for row in rows]
        return {
            "headers": headers,
            "rows": values,
            "page": page,
            "page_size": page_size,
            "row_count": metadata["row_count"],
            "indexing": False,
        }

    def close(self) -> None:
        with self.lock:
            self.connection.close()


def preview(policy: PathPolicy, raw_path: str, offset: int = 0, limit: int = MAX_PREVIEW_BYTES) -> dict[str, Any]:
    path = policy.resolve(raw_path)
    stat = path.stat()
    limit = max(1, min(MAX_PREVIEW_BYTES, limit))
    offset = max(0, min(offset, stat.st_size))
    kind = viewer_kind(path)
    result: dict[str, Any] = {
        "path": policy.relative(path),
        "kind": kind,
        "size": stat.st_size,
        "offset": offset,
        "truncated": offset + limit < stat.st_size,
    }
    if kind not in {"json", "text", "markdown", "html"}:
        return result
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(limit)
    result["bytes_read"] = len(data)
    result["next_offset"] = offset + len(data)
    text = data.decode("utf-8", errors="replace")
    if kind == "json" and offset == 0 and not result["truncated"]:
        try:
            result["json"] = json.loads(text)
        except json.JSONDecodeError:
            result["text"] = text
    else:
        result["text"] = text
    return result


def content_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
