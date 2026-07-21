#!/usr/bin/env python3
"""Tracked, digest-bound review state for research-journal relationships."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
LEDGER_NAME = "relationship-reviews.jsonl"
VERDICTS = {"typed", "proposed", "keep-related", "not-related", "deferred"}
RELATION_TYPES = {"supersedes", "contradicts", "refines", "complements", "supports", "answers"}
DIGEST_RE = re.compile(r"[0-9a-f]{64}")


class LedgerError(ValueError):
    """Raised when the review ledger is malformed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def note_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def durable_note_paths(root: Path, notebook: Path) -> list[Path]:
    paths: list[Path] = []
    for kind in ("learnings", "questions", "decisions"):
        directory = notebook / "notes" / kind
        if directory.is_dir():
            paths.extend(path.resolve() for path in directory.glob("*.md") if path.name != "README.md")
    return sorted(paths)


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise LedgerError(f"path is outside project root: {path}") from error


def pair_key(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise LedgerError("relationship review pair must contain two different notes")
    return tuple(sorted((left, right)))


class RelationshipReviews:
    def __init__(self, root: Path, notebook: str | Path = "lab-notebook") -> None:
        self.root = root.resolve()
        notebook_path = Path(notebook)
        self.notebook = (self.root / notebook_path).resolve() if not notebook_path.is_absolute() else notebook_path.resolve()
        self.path = self.notebook / LEDGER_NAME

    def ensure(self) -> bool:
        if self.path.exists():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        return True

    def note_paths(self) -> list[Path]:
        return durable_note_paths(self.root, self.notebook)

    def resolve_note(self, raw: str | Path) -> Path:
        path = Path(raw)
        candidate = path.resolve() if path.is_absolute() else (self.root / path).resolve()
        relative_path(candidate, self.root)
        if not candidate.is_file():
            raise LedgerError(f"note does not exist: {raw}")
        if candidate not in set(self.note_paths()):
            raise LedgerError(f"not a durable learning, question, or decision note: {raw}")
        return candidate

    def _validate_event(self, event: Any, line_number: int) -> dict[str, Any]:
        label = f"{self.path}:{line_number}"
        if not isinstance(event, dict):
            raise LedgerError(f"{label}: event must be a JSON object")
        if event.get("schema") != SCHEMA_VERSION:
            raise LedgerError(f"{label}: unsupported schema {event.get('schema')!r}")
        kind = event.get("event")
        if kind not in {"note-audited", "pair-reviewed"}:
            raise LedgerError(f"{label}: unsupported event {kind!r}")
        for key in ("timestamp", "reviewer"):
            if not isinstance(event.get(key), str) or not event[key].strip():
                raise LedgerError(f"{label}: {key} must be a non-empty string")
        def validate_path(raw: Any) -> None:
            if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
                raise LedgerError(f"{label}: paths must be project-relative and cannot escape the project")

        def validate_digest(raw: Any) -> None:
            if not isinstance(raw, str) or not DIGEST_RE.fullmatch(raw):
                raise LedgerError(f"{label}: digest must be a lowercase SHA-256 value")

        if kind == "note-audited":
            if not isinstance(event.get("path"), str) or not isinstance(event.get("digest"), str):
                raise LedgerError(f"{label}: note audit requires path and digest")
            validate_path(event["path"])
            validate_digest(event["digest"])
        else:
            paths = event.get("paths")
            digests = event.get("digests")
            if not isinstance(paths, list) or len(paths) != 2 or paths != sorted(paths) or paths[0] == paths[1]:
                raise LedgerError(f"{label}: pair review paths must be two distinct sorted paths")
            if not isinstance(digests, dict) or set(digests) != set(paths):
                raise LedgerError(f"{label}: pair review digests must match paths")
            for raw_path in paths:
                validate_path(raw_path)
                validate_digest(digests[raw_path])
            if event.get("verdict") not in VERDICTS:
                raise LedgerError(f"{label}: unsupported verdict {event.get('verdict')!r}")
            relationship = event.get("relationship")
            if relationship is not None:
                if not isinstance(relationship, dict) or not all(
                    isinstance(relationship.get(key), str) and relationship[key].strip()
                    for key in ("type", "source", "target", "reason")
                ):
                    raise LedgerError(f"{label}: relationship must contain type, source, target, and reason")
                if {relationship["source"], relationship["target"]} != set(paths):
                    raise LedgerError(f"{label}: relationship endpoints must match reviewed paths")
                if relationship["type"] not in RELATION_TYPES:
                    raise LedgerError(f"{label}: unsupported relationship type {relationship['type']!r}")
        return event

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        result: list[dict[str, Any]] = []
        for line_number, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as error:
                raise LedgerError(f"{self.path}:{line_number}: invalid JSON: {error.msg}") from error
            result.append(self._validate_event(event, line_number))
        return result

    def append(self, event: dict[str, Any]) -> None:
        self.ensure()
        self._validate_event(event, 1)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")

    def record_note_audit(self, path: Path, reviewer: str, note: str = "") -> dict[str, Any]:
        resolved = self.resolve_note(path)
        event = {
            "schema": SCHEMA_VERSION,
            "event": "note-audited",
            "timestamp": utc_now(),
            "reviewer": reviewer,
            "path": relative_path(resolved, self.root),
            "digest": note_digest(resolved),
        }
        if note:
            event["note"] = note
        self.append(event)
        return event

    def record_pair(
        self,
        left: Path,
        right: Path,
        verdict: str,
        reviewer: str,
        note: str = "",
        relationship: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if verdict not in VERDICTS:
            raise LedgerError(f"unsupported verdict: {verdict}")
        left_path, right_path = self.resolve_note(left), self.resolve_note(right)
        left_relative, right_relative = pair_key(
            relative_path(left_path, self.root), relative_path(right_path, self.root)
        )
        resolved = {relative_path(left_path, self.root): left_path, relative_path(right_path, self.root): right_path}
        event: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "event": "pair-reviewed",
            "timestamp": utc_now(),
            "reviewer": reviewer,
            "paths": [left_relative, right_relative],
            "digests": {
                left_relative: note_digest(resolved[left_relative]),
                right_relative: note_digest(resolved[right_relative]),
            },
            "verdict": verdict,
        }
        if note:
            event["note"] = note
        if relationship:
            event["relationship"] = relationship
        self.append(event)
        return event

    def latest_note_events(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for event in self.events():
            if event["event"] == "note-audited":
                latest[event["path"]] = event
        return latest

    def latest_pair_events(self) -> dict[tuple[str, str], dict[str, Any]]:
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for event in self.events():
            if event["event"] == "pair-reviewed":
                latest[tuple(event["paths"])] = event
        return latest

    def current_pair_reviews(self) -> dict[tuple[str, str], dict[str, Any]]:
        current: dict[tuple[str, str], dict[str, Any]] = {}
        for key, event in self.latest_pair_events().items():
            valid = True
            for raw_path in event["paths"]:
                path = self.root / raw_path
                if not path.is_file() or note_digest(path) != event["digests"].get(raw_path):
                    valid = False
                    break
            if valid:
                current[key] = event
        return current

    def status(self) -> dict[str, Any]:
        notes = {relative_path(path, self.root): path for path in self.note_paths()}
        latest_notes = self.latest_note_events()
        current_notes: list[str] = []
        stale_notes: list[str] = []
        unreviewed_notes: list[str] = []
        for raw_path, path in notes.items():
            event = latest_notes.get(raw_path)
            if not event:
                unreviewed_notes.append(raw_path)
            elif event["digest"] == note_digest(path):
                current_notes.append(raw_path)
            else:
                stale_notes.append(raw_path)
        latest_pairs = self.latest_pair_events()
        current_pairs = self.current_pair_reviews()
        stale_pairs = sorted([" :: ".join(key) for key in latest_pairs if key not in current_pairs])
        verdicts = Counter(event["verdict"] for event in current_pairs.values())
        last_audit = max(
            (event for event in self.events() if event["event"] == "note-audited"),
            key=lambda event: event["timestamp"],
            default=None,
        )
        return {
            "ledger": relative_path(self.path, self.root),
            "notes": {
                "total": len(notes),
                "current": len(current_notes),
                "stale": len(stale_notes),
                "unreviewed": len(unreviewed_notes),
                "current_paths": sorted(current_notes),
                "stale_paths": sorted(stale_notes),
                "unreviewed_paths": sorted(unreviewed_notes),
            },
            "pairs": {
                "total": len(latest_pairs),
                "current": len(current_pairs),
                "stale": len(stale_pairs),
                "stale_pairs": stale_pairs,
                "by_verdict": dict(sorted(verdicts.items())),
            },
            "last_audit": last_audit,
        }

    def audit_many(self, paths: Iterable[Path], reviewer: str, note: str = "") -> list[dict[str, Any]]:
        return [self.record_note_audit(path, reviewer, note) for path in paths]
