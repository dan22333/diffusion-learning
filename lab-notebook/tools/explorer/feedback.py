from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .model import utc_now


def default_state_dir(root: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-path", "research-journal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        candidate = Path(result.stdout.strip())
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve()
    except (OSError, subprocess.SubprocessError):
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        digest = uuid.uuid5(uuid.NAMESPACE_URL, root.resolve().as_uri()).hex[:16]
        return base / "research-journal" / digest


class FeedbackStore:
    def __init__(self, root: Path, state_dir: Path | None = None) -> None:
        self.root = root.resolve()
        self.state_dir = (state_dir or default_state_dir(root)).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "feedback.jsonl"
        self.lock = threading.RLock()

    def _events(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        events: list[dict[str, Any]] = []
        with self.lock:
            for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        events.append(event)
                except json.JSONDecodeError:
                    continue
        return events

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        event = {**event, "event_id": uuid.uuid4().hex, "timestamp": utc_now()}
        encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.lock:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(descriptor, encoded.encode("utf-8"))
            finally:
                os.close(descriptor)
        return event

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        target = payload.get("target")
        if not isinstance(target, dict) or not isinstance(target.get("path"), str):
            raise ValueError("feedback target.path is required")
        path = (self.root / target["path"]).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("feedback target escapes project root") from error
        feedback_id = uuid.uuid4().hex[:16]
        event = self.append(
            {
                "event": "created",
                "feedback_id": feedback_id,
                "kind": str(payload.get("kind") or "comment"),
                "message": str(payload.get("message") or "").strip(),
                "requested_action": str(payload.get("requested_action") or "").strip(),
                "target": target,
            }
        )
        return self._fold([event])[feedback_id]

    def update(self, feedback_id: str, status: str, note: str = "") -> dict[str, Any]:
        if status not in {"open", "claimed", "resolved", "dismissed"}:
            raise ValueError("invalid feedback status")
        self.append({"event": "status", "feedback_id": feedback_id, "status": status, "note": note})
        records = self.records()
        if feedback_id not in records:
            raise KeyError(feedback_id)
        return records[feedback_id]

    @staticmethod
    def _fold(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for event in events:
            feedback_id = event.get("feedback_id")
            if not isinstance(feedback_id, str):
                continue
            if event.get("event") == "created":
                records[feedback_id] = {
                    "id": feedback_id,
                    "created_at": event.get("timestamp"),
                    "updated_at": event.get("timestamp"),
                    "status": "open",
                    "kind": event.get("kind"),
                    "message": event.get("message"),
                    "requested_action": event.get("requested_action"),
                    "target": event.get("target"),
                    "history": [],
                }
            elif feedback_id in records:
                records[feedback_id]["status"] = event.get("status", records[feedback_id]["status"])
                records[feedback_id]["updated_at"] = event.get("timestamp")
                records[feedback_id]["history"].append(event)
        return records

    def records(self) -> dict[str, dict[str, Any]]:
        return self._fold(self._events())

    @staticmethod
    def _line_for_offset(text: str, offset: int) -> int:
        return text.count("\n", 0, max(0, offset)) + 1

    def resolve_target(self, record: dict[str, Any], documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
        target = record.get("target") or {}
        path = str(target.get("path") or "")
        relation = target.get("relation") or {}
        pair = relation.get("pair") or []
        if isinstance(pair, list) and len(pair) == 2:
            missing = [item for item in pair if not isinstance(item, str) or item not in documents]
            if missing:
                return {"attachment": "orphaned", "reason": "one or more relationship notes are unavailable", "kind": "relationship-pair"}
            return {"attachment": "attached", "kind": "relationship-pair", "pair": pair}
        document = documents.get(path)
        if not document:
            artifact = (self.root / path).resolve()
            try:
                artifact.relative_to(self.root)
            except ValueError:
                return {"attachment": "orphaned", "reason": "target file is unavailable"}
            if artifact.is_file():
                return {"attachment": "attached", "line_start": (target.get("snapshot_lines") or [1])[0], "kind": "file"}
            return {"attachment": "orphaned", "reason": "target file is unavailable"}
        selector = target.get("text_quote") or {}
        exact = str(selector.get("exact") or "")
        text = document.get("text", "")
        if exact:
            positions = list(find_literal_positions(exact, text))
            if positions:
                prefix = str(selector.get("prefix") or "")
                suffix = str(selector.get("suffix") or "")
                best = max(
                    positions,
                    key=lambda position: int(not prefix or text[max(0, position - len(prefix)):position].endswith(prefix))
                    + int(not suffix or text[position + len(exact):position + len(exact) + len(suffix)].startswith(suffix)),
                )
                line = self._line_for_offset(text, best)
                old_lines = target.get("snapshot_lines") or []
                moved = bool(old_lines and int(old_lines[0]) != line)
                return {"attachment": "moved" if moved else "attached", "line_start": line, "line_end": line + exact.count("\n")}
        heading_path = target.get("heading_path") or []
        expected = exact or str(target.get("quote") or "")
        candidates = [
            block
            for block in document.get("blocks", [])
            if not heading_path or block.get("heading_path", [])[-len(heading_path):] == heading_path
        ]
        if expected and candidates:
            best = max(candidates, key=lambda block: SequenceMatcher(None, expected, block.get("plain", "")).ratio())
            ratio = SequenceMatcher(None, expected, best.get("plain", "")).ratio()
            if ratio >= 0.68:
                return {
                    "attachment": "moved",
                    "line_start": best["line_start"],
                    "line_end": best["line_end"],
                    "similarity": round(ratio, 3),
                }
        return {"attachment": "orphaned", "reason": "selected text no longer matches"}

    def list_resolved(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        documents = {document["path"]: document for document in snapshot.get("documents", [])}
        records = self.records()
        result: list[dict[str, Any]] = []
        for record in records.values():
            result.append({**record, "resolution": self.resolve_target(record, documents)})
        return sorted(result, key=lambda item: item.get("updated_at") or "", reverse=True)


def find_literal_positions(needle: str, haystack: str):
    start = 0
    while True:
        position = haystack.find(needle, start)
        if position < 0:
            return
        yield position
        start = position + max(1, len(needle))
