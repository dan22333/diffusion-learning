from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import sqlite3
import threading
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from relationship_reviews import LedgerError, RelationshipReviews, pair_key


FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
FIELD_RE = re.compile(r"(?m)^{name}:\s*(.+?)\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
JOURNAL_DATE_RE = re.compile(r"^##\s+(20\d{2}-\d{2}-\d{2})(?:\s+[—-]\s+(.+?))?(?:\s+(\(backfilled\)))?\s*$", re.IGNORECASE)
JOURNAL_EVENT_RE = re.compile(r"^###\s+(.+?)\s*$")
TODO_ITEM_RE = re.compile(r"^\s*[-*]\s+\[\s\]\s+(.+?)\s*$")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
RELATION_RE = re.compile(
    r"^\s*[-*]\s+(supersedes|contradicts|refines|complements|supports|answers)\s*:\s*"
    r"(?:\[([^\]]+)\]\(([^)]+)\)|\[\[([^\]]+)\]\])(?:\s*[—-]\s*(.*))?\s*$",
    re.IGNORECASE,
)
RELATION_TYPES = {"supersedes", "contradicts", "refines", "complements", "supports", "answers"}
SECTION_EDGE_TYPES = {
    "source": "source",
    "sources": "source",
    "durable notes": "durable-note",
    "supersedes": "supersedes",
    "resolution": "answers",
    "related": "related",
}
PRIVATE_PARTS = {".extraction"}
STOPWORDS = {
    "about", "after", "again", "against", "also", "because", "being", "between", "does", "from", "have", "into",
    "more", "most", "only", "over", "should", "than", "that", "their", "then", "there", "these", "this", "those",
    "under", "using", "what", "when", "where", "which", "with", "would", "judge", "finding", "question", "decision",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = WIKI_LINK_RE.sub(r"\1", text)
    text = re.sub(r"[*_>#~-]", " ", text)
    return normalize(html.unescape(text))


def display_text(text: str) -> str:
    """Remove inline Markdown without flattening research punctuation."""
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = WIKI_LINK_RE.sub(r"\1", text)
    text = re.sub(r"(?<!\w)[*_]{1,2}|[*_]{1,2}(?!\w)", "", text)
    return normalize(html.unescape(text))


def parse_frontmatter(text: str) -> dict[str, str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    result: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip("\"'")
    return result


def field(text: str, name: str) -> str | None:
    match = re.search(FIELD_RE.pattern.format(name=re.escape(name)), text)
    return match.group(1).strip() if match else None


def first_heading(text: str, fallback: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    return match.group(1).strip() if match else fallback


def section_text(text: str, name: str) -> str:
    lines = text.splitlines()
    target = f"## {name}".lower()
    start = None
    for index, line in enumerate(lines):
        if line.strip().lower() == target:
            start = index + 1
            break
    if start is None:
        return ""
    result: list[str] = []
    for line in lines[start:]:
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) <= 2:
            break
        result.append(line)
    return "\n".join(result).strip()


def valid_date(value: str) -> str:
    match = re.fullmatch(r"20\d{2}-\d{2}-\d{2}", value.strip())
    if not match:
        return ""
    try:
        datetime.fromisoformat(match.group(0))
    except ValueError:
        return ""
    return match.group(0)


def tagged_journal_text(text: str, tag: str, following: tuple[str, ...]) -> str:
    stops = "|".join(re.escape(item) for item in following)
    pattern = rf"(?:^|\s){re.escape(tag)}(?:\s+\(intent vs result\))?:\s*(.*?)(?=(?:\s+(?:{stops})(?:\s+\(intent vs result\))?:)|$)"
    match = re.search(pattern, normalize(text), re.IGNORECASE)
    return display_text(match.group(1)) if match else ""


def parse_todo_groups(text: str) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current = "Uncategorized"
    for line in text.splitlines():
        heading = re.match(r"^###\s+(.+?)\s*$", line)
        if heading:
            current = display_text(heading.group(1))
            continue
        if not TODO_ITEM_RE.match(line):
            continue
        if not groups or groups[-1]["title"] != current:
            groups.append({"title": current, "count": 0})
        groups[-1]["count"] += 1
    return groups


def journal_blocks(text: str) -> list[tuple[str, str, str, bool, int]]:
    """Return dated journal event blocks in source order."""
    lines = text.splitlines()
    date_starts: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(lines):
        match = JOURNAL_DATE_RE.match(line)
        if match:
            date_starts.append((index, match))
    blocks: list[tuple[str, str, str, bool, int]] = []
    ordinal = 0
    for group_index, (start, date_match) in enumerate(date_starts):
        end = date_starts[group_index + 1][0] if group_index + 1 < len(date_starts) else len(lines)
        date_value = date_match.group(1)
        header_title = (date_match.group(2) or "").strip()
        header_backfilled = bool(date_match.group(3)) or "backfilled" in header_title.lower()
        body_lines = lines[start + 1 : end]
        event_starts = [(index, JOURNAL_EVENT_RE.match(line)) for index, line in enumerate(body_lines)]
        event_starts = [(index, match) for index, match in event_starts if match]
        if header_title:
            blocks.append((date_value, header_title, "\n".join(body_lines).strip(), header_backfilled, ordinal))
            ordinal += 1
            continue
        for event_index, (relative_start, event_match) in enumerate(event_starts):
            relative_end = event_starts[event_index + 1][0] if event_index + 1 < len(event_starts) else len(body_lines)
            title = event_match.group(1).strip()
            body = "\n".join(body_lines[relative_start + 1 : relative_end]).strip()
            blocks.append((date_value, title, body, header_backfilled or "backfilled" in title.lower(), ordinal))
            ordinal += 1
    return blocks


def classify_path(root: Path, notebook: Path, path: Path, experiment_roots: Iterable[Path] = ()) -> str:
    if path == notebook / "ACTIVE.md":
        return "active"
    if path == notebook / "JOURNAL.md":
        return "journal"
    if path == notebook / "TODO.md":
        return "todo"
    parts = path.parts
    if "notes" in parts:
        index = parts.index("notes")
        if index + 1 < len(parts):
            return {"learnings": "learning", "questions": "question", "decisions": "decision", "meetings": "meeting"}.get(parts[index + 1], "note")
    if path.parent == notebook / "sessions":
        return "session"
    if path.parent == notebook / "studies":
        return "study"
    if any(path == experiment or experiment in path.parents for experiment in experiment_roots):
        return "experiment" if path.name == "README.md" else "experiment-notes"
    return "note"


def parse_blocks(path: str, text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    headings: list[tuple[int, str]] = []
    blocks: list[dict[str, Any]] = []
    pending: list[str] = []
    pending_start = 1
    in_fence = False

    def flush(end_line: int) -> None:
        nonlocal pending
        markdown = "\n".join(pending).strip("\n")
        pending = []
        if not markdown.strip():
            return
        heading_path = [title for _, title in headings]
        plain = strip_markdown(markdown)
        identity = f"{path}\0{'/'.join(heading_path)}\0{normalize(plain)}"
        blocks.append(
            {
                "id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                "heading_path": heading_path,
                "markdown": markdown,
                "plain": plain,
                "line_start": pending_start,
                "line_end": end_line,
            }
        )

    for number, line in enumerate(lines, start=1):
        heading = HEADING_RE.match(line) if not in_fence else None
        if heading:
            flush(number - 1)
            level, title = len(heading.group(1)), heading.group(2).strip()
            headings = [(old_level, old_title) for old_level, old_title in headings if old_level < level]
            headings.append((level, title))
            if level > 1:
                pending_start = number
                pending.append(line)
            continue
        if line.strip().startswith("```"):
            if not pending:
                pending_start = number
            pending.append(line)
            in_fence = not in_fence
            continue
        if not in_fence and not line.strip():
            flush(number - 1)
            continue
        if not pending:
            pending_start = number
        pending.append(line)
    flush(len(lines))
    return blocks


def viewer_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".csv"}:
        return "table"
    if suffix == ".json":
        return "json"
    if suffix in {".jsonl", ".log", ".txt", ".py", ".yaml", ".yml", ".toml"}:
        return "text"
    if suffix == ".md":
        return "markdown"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in {".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"}:
        return "office"
    return "binary"


@dataclass
class ResearchModel:
    root: Path
    notebook_name: str = "lab-notebook"
    experiments_glob: str = "experiments/*"

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.notebook = (self.root / self.notebook_name).resolve()

    def experiment_directories(self) -> list[Path]:
        return sorted(path.resolve() for path in self.root.glob(self.experiments_glob) if path.is_dir())

    def document_paths(self, experiment_roots: Iterable[Path] | None = None) -> list[Path]:
        paths: set[Path] = set()
        for name in ("ACTIVE.md", "JOURNAL.md", "TODO.md"):
            path = self.notebook / name
            if path.is_file():
                paths.add(path)
        for kind in ("learnings", "questions", "decisions", "meetings"):
            directory = self.notebook / "notes" / kind
            if directory.is_dir():
                paths.update(path for path in directory.glob("*.md") if path.name != "README.md")
        sessions = self.notebook / "sessions"
        if sessions.is_dir():
            paths.update(sessions.glob("*.md"))
        studies = self.notebook / "studies"
        if studies.is_dir():
            paths.update(studies.glob("*.md"))
        for directory in experiment_roots if experiment_roots is not None else self.experiment_directories():
            for name in ("README.md", "notes.md"):
                path = directory / name
                if path.is_file():
                    paths.add(path)
        return sorted(path.resolve() for path in paths)

    def artifact_paths(self, experiment_roots: Iterable[Path] | None = None) -> list[Path]:
        experiment_roots = list(experiment_roots) if experiment_roots is not None else self.experiment_directories()
        paths: list[Path] = []
        document_paths = set(self.document_paths(experiment_roots))
        for directory in experiment_roots:
            for path in directory.rglob("*"):
                if path.is_file() and path.resolve() not in document_paths and not PRIVATE_PARTS.intersection(path.parts):
                    paths.append(path.resolve())
        return sorted(paths)

    def watched_state(self) -> dict[str, tuple[int, int]]:
        experiment_roots = self.experiment_directories()
        state: dict[str, tuple[int, int]] = {}
        for path in [*self.document_paths(experiment_roots), *self.artifact_paths(experiment_roots)]:
            try:
                stat = path.stat()
                state[path.as_posix()] = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                continue
        ledger = self.notebook / "relationship-reviews.jsonl"
        if ledger.is_file():
            stat = ledger.stat()
            state[ledger.as_posix()] = (stat.st_mtime_ns, stat.st_size)
        return state

    def _document(self, path: Path, experiment_roots: Iterable[Path] = ()) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8", errors="replace")
        meta = parse_frontmatter(text)
        relative = path.relative_to(self.root).as_posix()
        note_type = classify_path(self.root, self.notebook, path, experiment_roots)
        status = meta.get("status") or field(text, "Status") or "unknown"
        created = meta.get("created") or field(text, "Created") or field(text, "Date") or "unknown"
        updated = meta.get("updated") or meta.get("created") or field(text, "Date") or field(text, "Created") or "unknown"
        confidence = strip_markdown(section_text(text, "Confidence"))
        summary_heading = {
            "learning": "Claim",
            "question": "Question",
            "decision": "Decision",
            "experiment": "Question",
            "session": "Intent",
            "study": "Objective",
        }.get(note_type, "")
        summary = strip_markdown(section_text(text, summary_heading)) if summary_heading else ""
        return {
            "id": relative,
            "path": relative,
            "absolute_path": path.as_posix(),
            "type": note_type,
            "kind": meta.get("kind") or note_type,
            "title": first_heading(text, path.stem),
            "status": status,
            "created": created,
            "updated": updated,
            "confidence": confidence,
            "summary": summary[:400],
            "text": text,
            "blocks": parse_blocks(relative, text),
        }

    def _resolve_target(self, source: Path, raw: str, known: dict[str, str], stem_map: dict[str, list[str]]) -> str | None:
        raw = urllib.parse.unquote(raw.strip().split("#", 1)[0])
        if not raw or re.match(r"^[a-z]+://", raw, re.IGNORECASE) or raw.startswith("mailto:"):
            return None
        if raw.startswith("[[") and raw.endswith("]]" ):
            raw = raw[2:-2].split("|", 1)[0].split("#", 1)[0]
        candidate = (source.parent / raw).resolve()
        if candidate.is_dir():
            candidate = candidate / "README.md"
        try:
            relative = candidate.relative_to(self.root).as_posix()
        except ValueError:
            relative = ""
        if relative in known:
            return relative
        key = Path(raw).stem.lower()
        matches = stem_map.get(key, [])
        return matches[0] if len(matches) == 1 else None

    def _edges(self, documents: list[dict[str, Any]]) -> list[dict[str, str]]:
        known = {document["path"]: document["id"] for document in documents}
        stem_map: dict[str, list[str]] = {}
        for document in documents:
            stem_map.setdefault(Path(document["path"]).stem.lower(), []).append(document["id"])
        edges: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for document in documents:
            source_path = self.root / document["path"]
            current_section = ""
            for line in document["text"].splitlines():
                heading = HEADING_RE.match(line)
                if heading:
                    if len(heading.group(1)) <= 2:
                        current_section = heading.group(2).strip().lower()
                    continue
                if line.strip().lower() == "source:":
                    current_section = "source"
                    continue
                if current_section == "source" and line.strip() and not line.lstrip().startswith(("-", "*")):
                    current_section = ""
                relation = RELATION_RE.match(line) if current_section == "relationships" else None
                if relation:
                    relation_type = relation.group(1).lower()
                    raw_target = relation.group(3) or relation.group(4) or ""
                    target = self._resolve_target(source_path, raw_target, known, stem_map)
                    if target:
                        key = (document["id"], target, relation_type)
                        if key not in seen:
                            seen.add(key)
                            edges.append({"source": document["id"], "target": target, "type": relation_type, "reason": relation.group(5) or ""})
                    continue
                edge_type = SECTION_EDGE_TYPES.get(current_section, "cites")
                for _label, raw_target in MD_LINK_RE.findall(line):
                    target = self._resolve_target(source_path, raw_target, known, stem_map)
                    if target:
                        key = (document["id"], target, edge_type)
                        if key not in seen:
                            seen.add(key)
                            edges.append({"source": document["id"], "target": target, "type": edge_type, "reason": ""})
                for raw_target in WIKI_LINK_RE.findall(line):
                    target = self._resolve_target(source_path, raw_target, known, stem_map)
                    if target:
                        key = (document["id"], target, edge_type)
                        if key not in seen:
                            seen.add(key)
                            edges.append({"source": document["id"], "target": target, "type": edge_type, "reason": ""})
        return edges

    @staticmethod
    def _term_counts(document: dict[str, Any]) -> Counter[str]:
        words = re.findall(r"[a-z0-9][a-z0-9_-]+", f"{document['title']} {document['summary']}".lower())
        return Counter(word for word in words if len(word) >= 4 and word not in STOPWORDS)

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        numerator = sum(value * right.get(term, 0.0) for term, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def _relationship_candidates(
        self,
        documents: list[dict[str, Any]],
        edges: list[dict[str, str]],
        reviewed_pairs: set[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        durable = [document for document in documents if document["type"] in {"learning", "question", "decision"}]
        sources: dict[str, set[str]] = {document["id"]: set() for document in durable}
        explicit: set[frozenset[str]] = set()
        related: set[frozenset[str]] = set()
        for edge in edges:
            if edge["source"] in sources and edge["type"] == "source":
                sources[edge["source"]].add(edge["target"])
            pair = frozenset((edge["source"], edge["target"]))
            if edge["type"] in RELATION_TYPES:
                explicit.add(pair)
            elif edge["type"] == "related":
                related.add(pair)

        counts = {document["id"]: self._term_counts(document) for document in durable}
        document_frequency: Counter[str] = Counter()
        for terms in counts.values():
            document_frequency.update(terms.keys())
        total = max(1, len(durable))
        inverse_document_frequency = {
            term: math.log((1 + total) / (1 + frequency)) + 1.0
            for term, frequency in document_frequency.items()
        }
        vectors: dict[str, dict[str, float]] = {}
        for document in durable:
            terms = counts[document["id"]]
            vectors[document["id"]] = {
                term: (1.0 + math.log(frequency)) * inverse_document_frequency[term]
                for term, frequency in terms.items()
            }

        source_frequency: Counter[str] = Counter()
        for values in sources.values():
            source_frequency.update(values)
        rare_source_limit = max(2, math.ceil(total * 0.10))
        eligible: list[dict[str, Any]] = []
        reviewed = 0
        possible = 0
        current_statuses = {"active", "supported", "tentative", "open", "in-progress"}
        inactive_statuses = {"superseded", "contradicted", "answered", "obsolete"}
        for index, left in enumerate(durable):
            for right in durable[index + 1 :]:
                pair = frozenset((left["id"], right["id"]))
                if pair in explicit:
                    continue
                possible += 1
                if pair_key(left["id"], right["id"]) in (reviewed_pairs or set()):
                    reviewed += 1
                    continue
                lexical = self._cosine(vectors[left["id"]], vectors[right["id"]])
                overlap = set(counts[left["id"]]) & set(counts[right["id"]])
                distinctive_terms = sorted(overlap, key=lambda term: (-inverse_document_frequency[term], term))[:6]
                shared_sources = sorted(
                    sources[left["id"]] & sources[right["id"]],
                    key=lambda source: (source_frequency[source], source),
                )
                rare_sources = [source for source in shared_sources if source_frequency[source] <= rare_source_limit]
                direct_related = pair in related
                type_pair = {left["type"], right["type"]}
                question_finding = "question" in type_pair and bool(type_pair & {"learning", "decision"})
                lifecycle_divergence = (
                    left["status"] in current_statuses and right["status"] in inactive_statuses
                ) or (
                    right["status"] in current_statuses and left["status"] in inactive_statuses
                )
                patterns: list[str] = []
                if direct_related and question_finding:
                    patterns.append("An open research question is already linked to a current finding or decision")
                if direct_related and lifecycle_divergence:
                    patterns.append("Already-related notes have different lifecycle states")
                if direct_related and lexical >= 0.24:
                    patterns.append("The notes are already linked and their claims use distinctive shared terms")
                if rare_sources and lexical >= 0.16:
                    patterns.append("The notes share uncommon research evidence")
                if not patterns:
                    continue
                priority = (
                    4 * int(question_finding and direct_related)
                    + 3 * int(lifecycle_divergence and direct_related)
                    + 2 * int(bool(rare_sources))
                    + int(direct_related)
                    + lexical
                )
                ordered = sorted((left["id"], right["id"]))
                candidate_id = hashlib.sha256("\0".join(ordered).encode()).hexdigest()[:16]
                eligible.append(
                    {
                        "id": candidate_id,
                        "a": ordered[0],
                        "b": ordered[1],
                        "evidence": {
                            "patterns": patterns,
                            "distinctive_terms": distinctive_terms,
                            "shared_sources": rare_sources[:4],
                        },
                        "_priority": priority,
                    }
                )
        eligible.sort(key=lambda item: (-item["_priority"], item["a"], item["b"]))
        items = [{key: value for key, value in item.items() if key != "_priority"} for item in eligible]
        return {
            "items": items,
            "counts": {
                "shown": min(10, len(items)),
                "eligible": len(eligible),
                "suppressed": max(0, len(eligible) - min(10, len(items))),
                "below_gate": max(0, possible - len(eligible) - reviewed),
                "reviewed": reviewed,
            },
        }

    def _relationship_maintenance(self, edges: list[dict[str, str]]) -> dict[str, Any]:
        reviews = RelationshipReviews(self.root, self.notebook)
        try:
            status = reviews.status()
            current = reviews.current_pair_reviews()
        except LedgerError as error:
            return {
                "error": str(error),
                "notes": {"total": 0, "current": 0, "stale": 0, "unreviewed": 0, "stale_paths": [], "unreviewed_paths": []},
                "pairs": {"current": 0, "stale": 0, "by_verdict": {}},
                "accepted_unapplied": [],
            }
        explicit = {
            (edge["source"], edge["target"], edge["type"], edge.get("reason", ""))
            for edge in edges
            if edge["type"] in RELATION_TYPES
        }
        accepted_unapplied: list[dict[str, Any]] = []
        for event in current.values():
            relationship = event.get("relationship")
            if event["verdict"] != "typed" or not relationship:
                continue
            key = (
                relationship["source"], relationship["target"], relationship["type"], relationship["reason"]
            )
            if key not in explicit:
                accepted_unapplied.append(
                    {"paths": event["paths"], "verdict": event["verdict"], "relationship": relationship}
                )
        return {**status, "accepted_unapplied": accepted_unapplied, "error": ""}

    @staticmethod
    def _relationship_summary(documents: list[dict[str, Any]], edges: list[dict[str, str]]) -> dict[str, Any]:
        durable = [document for document in documents if document["type"] in {"learning", "question", "decision"}]
        document_map = {document["id"]: document for document in documents}
        explicit = [edge for edge in edges if edge["type"] in RELATION_TYPES]
        by_type = Counter(edge["type"] for edge in explicit)
        notes_with_relationships = {edge["source"] for edge in explicit}
        related_pairs = {frozenset((edge["source"], edge["target"])) for edge in edges if edge["type"] == "related"}
        legacy_supersedes = sum(bool(section_text(document["text"], "Supersedes")) for document in durable)
        issues: list[dict[str, str]] = []
        current = {"active", "supported", "tentative", "open", "in-progress"}
        for edge in explicit:
            source, target = document_map.get(edge["source"]), document_map.get(edge["target"])
            if not source or not target:
                continue
            if edge["type"] == "answers" and target["type"] != "question":
                issues.append({"level": "error", "message": f"answers must target a question: {target['title']}"})
            if edge["type"] == "supersedes" and target["status"] != "superseded":
                issues.append({"level": "warning", "message": f"Superseded target is still {target['status']}: {target['title']}"})
            if edge["type"] == "contradicts" and source["status"] in current and target["status"] in current:
                issues.append({"level": "attention", "message": f"Open contradiction: {source['title']} / {target['title']}"})
            if not edge.get("reason"):
                issues.append({"level": "warning", "message": f"{edge['type']} relationship lacks a rationale: {source['title']}"})
        return {
            "explicit": len(explicit),
            "by_type": dict(sorted(by_type.items())),
            "durable_notes": len(durable),
            "notes_with_relationships": len(notes_with_relationships),
            "generic_related_pairs": len(related_pairs),
            "legacy_supersedes_sections": legacy_supersedes,
            "issues": issues,
        }

    def _journal_entries(self, documents: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        journal_path = self.notebook / "JOURNAL.md"
        if not journal_path.is_file():
            return "", []
        journal_id = journal_path.relative_to(self.root).as_posix()
        known = {document["id"]: document for document in documents}
        entries: list[dict[str, Any]] = []
        text = journal_path.read_text(encoding="utf-8", errors="replace")
        for date_value, raw_title, body, backfilled, ordinal in journal_blocks(text):
            record_match = re.search(r"(?:^|\n)Record:\s*\[([^\]]+)\]\(([^)]+)\)", body, re.IGNORECASE)
            record_path = ""
            record_label = ""
            record_document: dict[str, Any] | None = None
            if record_match:
                record_label = display_text(record_match.group(1))
                raw_path = urllib.parse.unquote(record_match.group(2).split("#", 1)[0].strip())
                candidate = (journal_path.parent / raw_path).resolve()
                if candidate.is_dir():
                    candidate = candidate / "README.md"
                try:
                    relative = candidate.relative_to(self.root).as_posix()
                except ValueError:
                    relative = ""
                if relative in known:
                    record_path = relative
                    record_document = known[relative]
            source_kind = str((record_document or {}).get("kind") or (record_document or {}).get("type") or "research")
            category = {
                "experiment": "experiment",
                "study": "study",
                "reading": "reading",
                "synthesis": "synthesis",
                "infrastructure": "infrastructure",
                "learning": "finding",
                "question": "question",
                "decision": "decision",
            }.get(source_kind, "research")
            intent = tagged_journal_text(body, "Intent", ("Did", "Outcome", "Record"))
            outcome = tagged_journal_text(body, "Outcome", ("Record",)) or display_text(body)
            title = re.sub(r"\s*\(backfilled\)\s*$", "", display_text(raw_title), flags=re.IGNORECASE)
            entries.append(
                {
                    "date": date_value,
                    "title": title,
                    "intent": intent[:500],
                    "outcome": outcome[:700],
                    "category": category,
                    "backfilled": backfilled,
                    "record_path": record_path,
                    "record_label": record_label,
                    "journal_path": journal_id,
                    "ordinal": ordinal,
                }
            )
        entries.sort(key=lambda item: (item["date"], item["ordinal"]), reverse=True)
        for entry in entries:
            entry.pop("ordinal", None)
        return journal_id, entries

    def _dashboard(
        self,
        documents: list[dict[str, Any]],
        edges: list[dict[str, str]],
        active_target: str,
        relationship_summary: dict[str, Any],
        relationship_maintenance: dict[str, Any],
        counts: dict[str, int],
        artifact_count: int,
        journal: tuple[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        by_id = {document["id"]: document for document in documents}
        active_document = next((document for document in documents if document["type"] == "active"), None)
        target_document = by_id.get(active_target)
        focus_path = str((active_document or {}).get("id") or "")
        focus = {
            "configured": target_document is not None,
            "target": active_target,
            "path": target_document["id"] if target_document else focus_path,
            "title": target_document["title"] if target_document else "No active research target",
            "detail": (
                target_document.get("summary") or target_document["path"]
                if target_document
                else "ACTIVE.md explicitly records Target: none."
            ),
            "type": target_document["type"] if target_document else "active",
            "status": target_document["status"] if target_document else "none",
            "updated": valid_date(field((active_document or {}).get("text", ""), "Updated") or ""),
        }

        journal_path, journal_entries = journal if journal is not None else self._journal_entries(documents)
        latest_research = next((entry for entry in journal_entries if entry["category"] != "infrastructure"), None)
        latest_journal_date = journal_entries[0]["date"] if journal_entries else ""

        todo_document = next((document for document in documents if document["type"] == "todo"), None)
        todo_groups = parse_todo_groups(todo_document["text"]) if todo_document else []
        todo_count = sum(group["count"] for group in todo_groups)

        attention: list[dict[str, Any]] = []
        contradictions = [edge for edge in edges if edge["type"] == "contradicts"]
        if contradictions:
            attention.append(
                {
                    "kind": "contradictions",
                    "tone": "critical",
                    "count": len(contradictions),
                    "label": "Unresolved contradictions",
                    "detail": "Explicit typed conflicts require adjudication.",
                    "route": "precedence",
                    "target": "",
                }
            )
        running = [
            document
            for document in documents
            if document["type"] == "experiment" and document["status"] in {"running", "in-progress"}
        ]
        if running:
            attention.append(
                {
                    "kind": "running-experiments",
                    "tone": "active",
                    "count": len(running),
                    "label": "Running experiments",
                    "detail": "Experiment cards explicitly marked as active work.",
                    "route": "",
                    "target": running[0]["id"] if len(running) == 1 else "",
                }
            )
        notes_debt = relationship_maintenance.get("notes") or {}
        pairs_debt = relationship_maintenance.get("pairs") or {}
        review_debt = (
            int(notes_debt.get("stale") or 0)
            + int(notes_debt.get("unreviewed") or 0)
            + int(pairs_debt.get("stale") or 0)
            + len(relationship_maintenance.get("accepted_unapplied") or [])
        )
        if relationship_maintenance.get("error"):
            review_debt += 1
        if review_debt:
            attention.append(
                {
                    "kind": "relationship-review",
                    "tone": "warning",
                    "count": review_debt,
                    "label": "Relationship review debt",
                    "detail": "Changed, unaudited, stale, or accepted-but-unapplied relationship records.",
                    "route": "precedence",
                    "target": "",
                }
            )
        if todo_count:
            attention.append(
                {
                    "kind": "todo",
                    "tone": "normal",
                    "count": todo_count,
                    "label": "Open research actions",
                    "detail": f"Recorded across {len(todo_groups)} TODO themes.",
                    "route": "",
                    "target": str((todo_document or {}).get("id") or ""),
                }
            )
        newer_experiments = [
            document
            for document in documents
            if document["type"] == "experiment"
            and valid_date(str(document.get("created") or ""))
            and (not latest_journal_date or document["created"] > latest_journal_date)
        ]
        if newer_experiments:
            attention.append(
                {
                    "kind": "journal-gap",
                    "tone": "warning",
                    "count": len(newer_experiments),
                    "label": "Journal coverage gap",
                    "detail": "Experiment cards are newer than the latest dated journal entry.",
                    "route": "",
                    "target": journal_path,
                }
            )

        def knowledge_group(document_type: str) -> dict[str, Any]:
            selected = [document for document in documents if document["type"] == document_type]
            statuses = Counter(str(document["status"]) for document in selected)
            return {"total": len(selected), "statuses": dict(sorted(statuses.items()))}

        return {
            "focus": focus,
            "latest_research": latest_research,
            "activity": journal_entries[:24],
            "journal_path": journal_path,
            "attention": attention,
            "todo": {"count": todo_count, "groups": todo_groups, "path": str((todo_document or {}).get("id") or "")},
            "knowledge": {
                "learnings": knowledge_group("learning"),
                "questions": knowledge_group("question"),
                "decisions": knowledge_group("decision"),
                "relationships": {
                    "explicit": int(relationship_summary.get("explicit") or 0),
                    "reviewed": int(notes_debt.get("current") or 0),
                    "total": int(notes_debt.get("total") or 0),
                },
            },
            "inventory": {**dict(sorted(counts.items())), "raw-artifact": artifact_count},
        }

    def build(self) -> dict[str, Any]:
        experiment_roots = self.experiment_directories()
        documents = [self._document(path, experiment_roots) for path in self.document_paths(experiment_roots)]
        document_ids = {document["id"] for document in documents}
        active_target = ""
        active_path = self.notebook / "ACTIVE.md"
        if active_path.is_file():
            raw_target = (field(active_path.read_text(encoding="utf-8", errors="replace"), "Target") or "").strip().strip("`")
            if raw_target.lower() not in {"", "none"}:
                candidate = (self.root / raw_target).resolve()
                if candidate.is_dir():
                    candidate = candidate / "README.md"
                try:
                    relative_target = candidate.relative_to(self.root).as_posix()
                except ValueError:
                    relative_target = ""
                if relative_target in document_ids:
                    active_target = relative_target
        artifacts: list[dict[str, Any]] = []
        for path in self.artifact_paths(experiment_roots):
            try:
                stat = path.stat()
            except OSError:
                continue
            relative = path.relative_to(self.root).as_posix()
            experiment_root = next((directory for directory in experiment_roots if path == directory or directory in path.parents), None)
            experiment = experiment_root.relative_to(self.root).as_posix() if experiment_root else ""
            artifacts.append(
                {
                    "id": relative,
                    "path": relative,
                    "name": path.name,
                    "experiment": experiment,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                    "kind": viewer_kind(path),
                }
            )
        edges = self._edges(documents)
        relationship_maintenance = self._relationship_maintenance(edges)
        try:
            reviewed_pairs = set(RelationshipReviews(self.root, self.notebook).current_pair_reviews())
        except LedgerError:
            reviewed_pairs = set()
        relationship_candidates = self._relationship_candidates(documents, edges, reviewed_pairs)
        relationship_summary = self._relationship_summary(documents, edges)
        counts: dict[str, int] = {}
        for document in documents:
            counts[document["type"]] = counts.get(document["type"], 0) + 1
        journal_path, journal_entries = self._journal_entries(documents)
        dashboard = self._dashboard(
            documents,
            edges,
            active_target,
            relationship_summary,
            relationship_maintenance,
            counts,
            len(artifacts),
            journal=(journal_path, journal_entries),
        )
        return {
            "generated_at": utc_now(),
            "project": self.root.name,
            "root": self.root.as_posix(),
            "active_target": active_target,
            "journal": {"path": journal_path, "entries": journal_entries},
            "documents": documents,
            "edges": edges,
            "relationship_candidates": relationship_candidates,
            "relationship_summary": relationship_summary,
            "relationship_maintenance": relationship_maintenance,
            "dashboard": dashboard,
            "artifacts": artifacts,
            "counts": counts,
        }


class SearchIndex:
    def __init__(self, database: Path) -> None:
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.fts_available = True
        self.connection = sqlite3.connect(database, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._schema()

    def _schema(self) -> None:
        with self.lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS search_blocks (id TEXT PRIMARY KEY, doc_id TEXT, path TEXT, title TEXT, heading TEXT, body TEXT, type TEXT, status TEXT)"
            )
            try:
                self.connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(id UNINDEXED, doc_id UNINDEXED, path UNINDEXED, title, heading, body, type UNINDEXED, status UNINDEXED)"
                )
            except sqlite3.OperationalError:
                self.fts_available = False
            self.connection.commit()

    def rebuild(self, snapshot: dict[str, Any]) -> None:
        rows: list[tuple[str, ...]] = []
        for document in snapshot["documents"]:
            for block in document["blocks"]:
                rows.append(
                    (
                        f"{document['id']}:{block['id']}",
                        document["id"],
                        document["path"],
                        document["title"],
                        " / ".join(block["heading_path"]),
                        block["plain"],
                        document["type"],
                        document["status"],
                    )
                )
        with self.lock:
            self.connection.execute("DELETE FROM search_blocks")
            self.connection.executemany("INSERT INTO search_blocks VALUES (?,?,?,?,?,?,?,?)", rows)
            if self.fts_available:
                self.connection.execute("DELETE FROM search_fts")
                self.connection.executemany("INSERT INTO search_fts VALUES (?,?,?,?,?,?,?,?)", rows)
            self.connection.commit()

    def search(self, query: str, limit: int = 80) -> list[dict[str, Any]]:
        terms = re.findall(r"[\w.-]+", query.lower(), flags=re.UNICODE)
        if not terms:
            return []
        with self.lock:
            if self.fts_available:
                expression = " AND ".join(f'"{term.replace(chr(34), chr(34) * 2)}"*' for term in terms)
                try:
                    rows = self.connection.execute(
                        "SELECT doc_id, path, title, heading, type, status, snippet(search_fts, 5, '<mark>', '</mark>', '…', 24) AS snippet, bm25(search_fts, 1.0, 1.0, 8.0, 4.0, 1.0) AS rank FROM search_fts WHERE search_fts MATCH ? ORDER BY rank LIMIT ?",
                        (expression, limit),
                    ).fetchall()
                    return [dict(row) for row in rows]
                except sqlite3.OperationalError:
                    pass
            clauses = " AND ".join("(lower(title) LIKE ? OR lower(heading) LIKE ? OR lower(body) LIKE ?)" for _ in terms)
            params: list[Any] = []
            for term in terms:
                params.extend([f"%{term}%"] * 3)
            params.append(limit)
            rows = self.connection.execute(
                f"SELECT doc_id, path, title, heading, type, status, substr(body, 1, 300) AS snippet, 0 AS rank FROM search_blocks WHERE {clauses} LIMIT ?",
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def close(self) -> None:
        with self.lock:
            self.connection.close()
