#!/usr/bin/env python3
"""Read-only-by-default integrity checks for a research-journal notebook."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

from index import JournalIndex
from relationship_reviews import LedgerError, RelationshipReviews, pair_key


ARTIFACT_KINDS = {"study", "experiment", "session", "reading", "infrastructure", "synthesis"}
ARTIFACT_STATUSES = {"planned", "running", "done", "partial", "abandoned", "superseded"}
QUESTION_STATUSES = {"open", "in-progress", "answered", "blocked", "obsolete"}
LEARNING_STATUSES = {"tentative", "supported", "contradicted", "superseded"}
DECISION_STATUSES = {"active", "superseded"}
EXPLORER_VERSION = "22"
MARKER_START = "<!-- RESEARCH-MODE:ACTIVE"
MARKER_END = "<!-- /RESEARCH-MODE -->"
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
RELATION_TYPES = {"supersedes", "contradicts", "refines", "complements", "supports", "answers"}
RELATION_RE = re.compile(r"^\s*[-*]\s+([a-z-]+)\s*:\s*\[[^\]]+\]\(([^)]+)\)(?:\s*[—-]\s*(.+?))?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Issue:
    level: str
    code: str
    path: str
    message: str


def parse_frontmatter(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return {}
    if not lines or lines[0].strip() != "---":
        return {}
    result: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return result
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip('"\'')
    return {}


def section_lines(text: str, heading: str) -> list[str] | None:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start = index + 1
            break
    if start is None:
        return None
    result: list[str] = []
    level = len(heading) - len(heading.lstrip("#"))
    for line in lines[start:]:
        match = re.match(r"^(#+)\s", line)
        if match and len(match.group(1)) <= level:
            break
        result.append(line)
    return result


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_markdown_link(source: Path, raw_target: str) -> Path | None:
    target = unquote(raw_target.split("#", 1)[0].strip())
    if not target or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
        return None
    candidate = (source.parent / target).resolve()
    if candidate.exists():
        return candidate
    if not candidate.suffix and candidate.with_suffix(".md").exists():
        return candidate.with_suffix(".md")
    return candidate


def field(text: str, name: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(name)}:\s*(.+?)\s*$", text)
    return match.group(1).strip() if match else None


class Doctor:
    def __init__(
        self,
        root: Path,
        notebook: str,
        experiments: str,
        sessions: str | None,
    ) -> None:
        self.root = root.resolve()
        self.notebook = (self.root / notebook).resolve()
        self.experiments_glob = experiments
        self.sessions = (self.root / (sessions or f"{notebook}/sessions")).resolve()
        self.index = JournalIndex(self.root, notebook, experiments, sessions)
        self.issues: list[Issue] = []
        self.artifacts: list[tuple[Path, dict[str, str]]] = []

    def add(self, level: str, code: str, path: Path, message: str) -> None:
        self.issues.append(Issue(level, code, relative(path, self.root), message))

    def run(self) -> list[Issue]:
        self.check_activation()
        self.check_core_files()
        self.check_artifacts()
        self.check_active_pointer()
        self.check_journal_freshness()
        self.check_durable_notes()
        self.check_relationship_graph()
        self.check_relationship_reviews()
        self.check_questions_and_todo()
        self.check_duplicate_ids()
        self.check_index()
        self.check_machine_json()
        return sorted(self.issues, key=lambda item: (item.level != "error", item.path, item.code))

    def instruction_files(self) -> list[Path]:
        files: dict[Path, Path] = {}
        for name in ("CLAUDE.md", "AGENTS.md"):
            path = self.root / name
            if path.exists() or path.is_symlink():
                try:
                    files[path.resolve()] = path
                except OSError:
                    self.add("error", "broken-instruction-link", path, "instruction-file symlink is broken")
        return list(files)

    def check_activation(self) -> None:
        files = self.instruction_files()
        active = 0
        for path in files:
            text = path.read_text(encoding="utf-8")
            starts = text.count(MARKER_START)
            ends = text.count(MARKER_END)
            if starts != ends:
                self.add("error", "unbalanced-activation", path, f"found {starts} starts and {ends} ends")
            if starts > 1:
                self.add("error", "duplicate-activation", path, f"found {starts} activation stanzas")
            active += starts
        if active == 0:
            self.add("warning", "research-mode-off", self.root, "no active research-mode stanza found")
        if active > len(files):
            self.add("error", "activation-count", self.root, "activation stanzas exceed unique instruction files")

    def check_core_files(self) -> None:
        required = (
            "AGENTS.md",
            "TEMPLATES.md",
            "EXPLORER.md",
            "JOURNAL.md",
            "TODO.md",
            "ACTIVE.md",
            "relationship-reviews.jsonl",
        )
        if not self.notebook.is_dir():
            self.add("error", "missing-notebook", self.notebook, "notebook directory is missing")
            return
        for name in required:
            path = self.notebook / name
            if not path.is_file():
                self.add("error", "missing-core-file", path, "required notebook file is missing")
        local_tools = (
            "doctor.py",
            "index.py",
            "relationship_reviews.py",
            "relationships.py",
            "explorer/server.py",
            "explorer/feedback_cli.py",
            "explorer/static/index.html",
            "explorer/static/app.js",
            "explorer/static/app.css",
            "explorer/VERSION",
        )
        for name in local_tools:
            path = self.notebook / "tools" / name
            if not path.is_file():
                self.add("warning", "missing-local-tool", path, "project-local notebook tool is missing")
        version_path = self.notebook / "tools" / "explorer" / "VERSION"
        if version_path.is_file() and version_path.read_text(encoding="utf-8").strip() != EXPLORER_VERSION:
            self.add("warning", "outdated-local-tool", version_path, f"expected explorer version {EXPLORER_VERSION}")
        if not self.sessions.is_dir():
            self.add("error", "missing-sessions-dir", self.sessions, "non-experiment sessions directory is missing")
        studies = self.notebook / "studies"
        if not studies.is_dir():
            self.add("error", "missing-studies-dir", studies, "multi-experiment studies directory is missing")

    def experiment_dirs(self) -> list[Path]:
        return sorted(path for path in self.root.glob(self.experiments_glob) if path.is_dir())

    def session_files(self) -> list[Path]:
        if not self.sessions.is_dir():
            return []
        return sorted(path for path in self.sessions.glob("*.md") if path.is_file())

    def study_files(self) -> list[Path]:
        studies = self.notebook / "studies"
        if not studies.is_dir():
            return []
        return sorted(path for path in studies.glob("*.md") if path.is_file())

    def check_artifact_metadata(self, path: Path, expected_kind: str | None = None) -> dict[str, str]:
        meta = parse_frontmatter(path)
        if not meta:
            self.add("error", "missing-frontmatter", path, "artifact requires YAML frontmatter")
            return {}
        for key in ("kind", "status", "created", "updated"):
            if not meta.get(key):
                self.add("error", "missing-metadata", path, f"missing frontmatter field: {key}")
        kind = meta.get("kind")
        status = meta.get("status")
        if kind and kind not in ARTIFACT_KINDS:
            self.add("error", "invalid-kind", path, f"invalid kind: {kind}")
        if expected_kind and kind and kind != expected_kind:
            self.add("error", "misrouted-artifact", path, f"expected kind {expected_kind}, found {kind}")
        if status and status not in ARTIFACT_STATUSES:
            self.add("error", "invalid-artifact-status", path, f"invalid status: {status}")
        for key in ("created", "updated"):
            value = meta.get(key)
            if value:
                try:
                    date.fromisoformat(value)
                except ValueError:
                    self.add("error", "invalid-date", path, f"{key} is not YYYY-MM-DD: {value}")
        return meta

    def check_artifacts(self) -> None:
        for study in self.study_files():
            meta = self.check_artifact_metadata(study, "study")
            self.artifacts.append((study, meta))
            self.check_artifact_sections(study, require_durable=True, require_extraction=False)
            text = study.read_text(encoding="utf-8")
            for heading in ("Objective", "Design commitments", "Experiment registry"):
                if section_lines(text, f"## {heading}") is None:
                    self.add("error", "missing-study-section", study, f"missing ## {heading} section")
        for directory in self.experiment_dirs():
            readme = directory / "README.md"
            notes = directory / "notes.md"
            if not readme.is_file():
                self.add("error", "missing-experiment-card", readme, "experiment README.md is missing")
                continue
            if not notes.is_file():
                self.add("error", "missing-experiment-notes", notes, "experiment notes.md is missing")
            meta = self.check_artifact_metadata(readme, "experiment")
            self.artifacts.append((readme, meta))
            self.check_artifact_sections(readme, require_durable=True, require_extraction=False)
            if notes.is_file():
                self.check_artifact_sections(notes, require_durable=False, require_extraction=True)
        for session in self.session_files():
            meta = self.check_artifact_metadata(session)
            if meta.get("kind") == "experiment":
                self.add("error", "misrouted-artifact", session, "experiment belongs in the experiment tree")
            self.artifacts.append((session, meta))
            self.check_artifact_sections(session, require_durable=True, require_extraction=True)

    def check_artifact_sections(self, path: Path, require_durable: bool, require_extraction: bool) -> None:
        text = path.read_text(encoding="utf-8")
        durable = section_lines(text, "## Durable notes")
        if require_durable and durable is None:
            self.add("error", "missing-durable-section", path, "missing ## Durable notes section")
        elif durable is not None:
            for line in durable:
                if not line.lstrip().startswith("-"):
                    continue
                links = LINK_RE.findall(line)
                if not links:
                    self.add("warning", "nonlink-durable-note", path, f"durable-note entry is not a Markdown link: {line.strip()}")
                    continue
                for raw in links:
                    target = resolve_markdown_link(path, raw)
                    if target is not None and not target.exists():
                        self.add("error", "broken-durable-link", path, f"link does not resolve: {raw}")
        if not require_extraction:
            return
        queue = section_lines(text, "## Things to extract later")
        if queue is None:
            self.add("error", "missing-extraction-section", path, "missing ## Things to extract later section")
            return
        for line in queue:
            stripped = line.strip()
            if not stripped.startswith("-"):
                continue
            if re.match(r"^- \[ \]", stripped):
                self.add("warning", "pending-extraction", path, stripped)
            elif re.match(r"^- \[[xX]\]", stripped):
                lowered = stripped.lower()
                if "extracted:" not in lowered and "resolved:" not in lowered:
                    self.add("error", "unresolved-completion", path, "checked item lacks extracted: or resolved:")
                for raw in LINK_RE.findall(stripped):
                    target = resolve_markdown_link(path, raw)
                    if target is not None and not target.exists():
                        self.add("error", "broken-extraction-link", path, f"link does not resolve: {raw}")
            else:
                self.add("error", "legacy-extraction-item", path, f"queue item lacks checkbox: {stripped}")

    def check_active_pointer(self) -> None:
        path = self.notebook / "ACTIVE.md"
        if not path.is_file():
            return
        target = field(path.read_text(encoding="utf-8"), "Target")
        if target is None:
            self.add("error", "invalid-active-pointer", path, "missing Target field")
            return
        if target.lower() == "none":
            return
        resolved = (self.root / target).resolve()
        if not resolved.exists():
            self.add("error", "broken-active-pointer", path, f"target does not exist: {target}")

    def check_journal_freshness(self) -> None:
        journal = self.notebook / "JOURNAL.md"
        if not journal.is_file():
            return
        dates = [date.fromisoformat(item) for item in DATE_RE.findall(journal.read_text(encoding="utf-8"))]
        if not dates:
            self.add("warning", "undated-journal", journal, "journal contains no YYYY-MM-DD date")
            return
        latest = max(dates)
        for path, meta in self.artifacts:
            created = meta.get("created")
            if not created:
                continue
            artifact_date = date.fromisoformat(created)
            if artifact_date > latest:
                self.add(
                    "warning",
                    "journal-gap",
                    path,
                    f"artifact date {artifact_date.isoformat()} is newer than journal {latest.isoformat()}",
                )

    def durable_note_files(self) -> Iterable[tuple[str, Path]]:
        for note_type in ("learnings", "questions", "decisions"):
            directory = self.notebook / "notes" / note_type
            if not directory.is_dir():
                self.add("error", "missing-note-dir", directory, "durable-note directory is missing")
                continue
            for path in sorted(directory.glob("*.md")):
                if path.name != "README.md":
                    yield note_type, path

    def source_lines(self, text: str) -> list[str]:
        lines = text.splitlines()
        start = None
        for index, line in enumerate(lines):
            if line.strip() == "Source:":
                start = index + 1
                break
        if start is None:
            return []
        result = []
        for line in lines[start:]:
            if line.startswith("#") or (line and not line.startswith((" ", "-"))):
                break
            if line.strip().startswith("-"):
                result.append(line.strip())
        return result

    def check_durable_notes(self) -> None:
        allowed = {
            "learnings": LEARNING_STATUSES,
            "questions": QUESTION_STATUSES,
            "decisions": DECISION_STATUSES,
        }
        required_sections = {
            "learnings": ("Claim", "Evidence", "Why it matters", "Confidence"),
            "questions": ("Question", "Why it matters", "Current evidence", "Next checks", "Resolution"),
            "decisions": ("Decision", "Reason", "Consequences"),
        }
        required_fields = {"learnings": ("Date",), "questions": ("Created",), "decisions": ("Date",)}
        for note_type, path in self.durable_note_files():
            text = path.read_text(encoding="utf-8")
            status = field(text, "Status")
            if status not in allowed[note_type]:
                self.add("error", "invalid-note-status", path, f"invalid or missing Status: {status}")
            for name in required_fields[note_type]:
                if not field(text, name):
                    self.add("error", "missing-note-field", path, f"missing {name}: field")
            for name in required_sections[note_type]:
                if section_lines(text, f"## {name}") is None:
                    self.add("error", "missing-note-section", path, f"missing ## {name} section")
            self.check_relationships(path, text)
            sources = self.source_lines(text)
            if not sources:
                self.add("error", "missing-source", path, "durable note has no Source entries")
            for source_line in sources:
                links = LINK_RE.findall(source_line)
                if not links:
                    if re.search(r"(?:experiments/|lab-notebook/(?:sessions|studies)/)", source_line):
                        self.add("warning", "nonlink-source", path, f"path-like source is not a Markdown link: {source_line}")
                    continue
                for raw in links:
                    target = resolve_markdown_link(path, raw)
                    if target is None:
                        continue
                    if not target.exists():
                        self.add("error", "broken-source-link", path, f"source link does not resolve: {raw}")
                        continue
                    source_file = target / "README.md" if target.is_dir() else target
                    if source_file.is_file() and path.stem not in source_file.read_text(encoding="utf-8"):
                        self.add("warning", "missing-reverse-link", path, f"source does not link back: {relative(source_file, self.root)}")

    def check_relationships(self, path: Path, text: str) -> None:
        lines = section_lines(text, "## Relationships")
        if lines is None:
            return
        for line in lines:
            if not line.strip() or not line.lstrip().startswith(("-", "*")):
                continue
            match = RELATION_RE.match(line)
            if not match:
                self.add("error", "invalid-relationship", path, f"relationship must be '<type>: [title](path)': {line.strip()}")
                continue
            relation_type, raw, reason = match.group(1).lower(), match.group(2), (match.group(3) or "").strip()
            if relation_type not in RELATION_TYPES:
                self.add("error", "invalid-relationship-type", path, f"invalid relationship type: {relation_type}")
            target = resolve_markdown_link(path, raw)
            if target is not None and not target.exists():
                self.add("error", "broken-relationship-link", path, f"relationship link does not resolve: {raw}")
            if target is not None and target.resolve() == path.resolve():
                self.add("error", "self-relationship", path, f"relationship points to its own note: {raw}")
            if not reason:
                self.add("warning", "missing-relationship-reason", path, f"{relation_type} relationship lacks a rationale: {raw}")

    def check_relationship_graph(self) -> None:
        notes: dict[Path, tuple[str, str]] = {}
        relations: list[tuple[Path, Path, str]] = []
        for note_type, path in self.durable_note_files():
            text = path.read_text(encoding="utf-8")
            notes[path.resolve()] = (note_type, field(text, "Status") or "")
            lines = section_lines(text, "## Relationships") or []
            for line in lines:
                match = RELATION_RE.match(line)
                if not match or match.group(1).lower() not in RELATION_TYPES:
                    continue
                target = resolve_markdown_link(path, match.group(2))
                if target is not None and target.exists():
                    relations.append((path.resolve(), target.resolve(), match.group(1).lower()))

        seen: set[tuple[Path, Path, str]] = set()
        for source, target, relation_type in relations:
            key = (source, target, relation_type)
            if key in seen:
                self.add("error", "duplicate-relationship", source, f"duplicate {relation_type} relationship to {relative(target, self.root)}")
            seen.add(key)
            target_type, target_status = notes.get(target, ("", ""))
            source_type, source_status = notes.get(source, ("", ""))
            if relation_type == "answers":
                if target_type != "questions":
                    self.add("error", "invalid-answers-target", source, f"answers must target a question: {relative(target, self.root)}")
                elif target_status != "answered":
                    self.add("warning", "unclosed-answered-question", source, f"answers target is still {target_status}: {relative(target, self.root)}")
            if relation_type == "supersedes" and target_status != "superseded":
                self.add("error", "active-superseded-target", source, f"supersedes target is still {target_status}: {relative(target, self.root)}")
            if relation_type == "contradicts" and source_status in {"active", "supported", "tentative"} and target_status in {"active", "supported", "tentative"}:
                self.add("warning", "open-contradiction", source, f"both contradictory notes remain current: {relative(target, self.root)}")

        incoming_supersedes = {target for _source, target, relation_type in relations if relation_type == "supersedes"}
        for path, (_note_type, status) in notes.items():
            if status == "superseded" and path not in incoming_supersedes:
                self.add("error", "unlinked-superseded-note", path, "superseded note has no incoming supersedes relationship")

        for relation_type in ("supersedes", "refines"):
            adjacency: dict[Path, set[Path]] = {}
            for source, target, edge_type in relations:
                if edge_type == relation_type:
                    adjacency.setdefault(source, set()).add(target)
            visiting: set[Path] = set()
            visited: set[Path] = set()

            def visit(node: Path) -> bool:
                if node in visiting:
                    return True
                if node in visited:
                    return False
                visiting.add(node)
                cyclic = any(visit(target) for target in adjacency.get(node, set()))
                visiting.remove(node)
                visited.add(node)
                return cyclic

            if any(visit(node) for node in list(adjacency)):
                self.add("error", "relationship-cycle", self.notebook, f"{relation_type} relationships contain a cycle")

    def check_relationship_reviews(self) -> None:
        reviews = RelationshipReviews(self.root, self.notebook)
        if not reviews.path.is_file():
            return
        try:
            status = reviews.status()
            current_pairs = reviews.current_pair_reviews()
        except LedgerError as error:
            self.add("error", "malformed-relationship-reviews", reviews.path, str(error))
            return
        notes = status["notes"]
        if notes["stale"]:
            self.add(
                "warning",
                "stale-relationship-audits",
                reviews.path,
                f"{notes['stale']} durable note(s) changed since their last semantic relationship audit",
            )
        if notes["unreviewed"]:
            self.add(
                "warning",
                "unaudited-relationships",
                reviews.path,
                f"{notes['unreviewed']} durable note(s) have never received a semantic relationship audit",
            )
        if status["pairs"]["stale"]:
            self.add(
                "warning",
                "stale-pair-reviews",
                reviews.path,
                f"{status['pairs']['stale']} pair decision(s) are stale because a reviewed note changed",
            )

        explicit: dict[tuple[str, str], set[tuple[str, str, str, str]]] = {}
        for _note_type, path in self.durable_note_files():
            text = path.read_text(encoding="utf-8")
            for line in section_lines(text, "## Relationships") or []:
                match = RELATION_RE.match(line)
                if not match or match.group(1).lower() not in RELATION_TYPES:
                    continue
                target = resolve_markdown_link(path, match.group(2))
                if target is None or not target.exists():
                    continue
                source_raw, target_raw = relative(path, self.root), relative(target, self.root)
                relationship = (
                    source_raw,
                    target_raw,
                    match.group(1).lower(),
                    (match.group(3) or "").strip(),
                )
                explicit.setdefault(pair_key(source_raw, target_raw), set()).add(relationship)

        for key, relationships in explicit.items():
            review = current_pairs.get(key)
            reviewed_relationship = review.get("relationship") if review and review.get("verdict") == "typed" else None
            reviewed_tuple = (
                reviewed_relationship.get("source"),
                reviewed_relationship.get("target"),
                reviewed_relationship.get("type"),
                reviewed_relationship.get("reason"),
            ) if reviewed_relationship else None
            if reviewed_tuple not in relationships:
                self.add(
                    "warning",
                    "unreviewed-explicit-relationship",
                    self.root / next(iter(relationships))[0],
                    "explicit relationship has no current matching typed review",
                )
        for key, review in current_pairs.items():
            relationship = review.get("relationship")
            if review.get("verdict") != "typed" or not relationship:
                continue
            reviewed_tuple = tuple(relationship[name] for name in ("source", "target", "type", "reason"))
            if reviewed_tuple not in explicit.get(key, set()):
                self.add(
                    "warning",
                    "accepted-relationship-unapplied",
                    reviews.path,
                    f"typed review is not present in its canonical source note: {relationship['source']} -> {relationship['target']}",
                )

    def question_files(self) -> list[Path]:
        directory = self.notebook / "notes" / "questions"
        return sorted(path for path in directory.glob("q*.md")) if directory.is_dir() else []

    def check_questions_and_todo(self) -> None:
        statuses: dict[str, str] = {}
        for path in self.question_files():
            text = path.read_text(encoding="utf-8")
            status = field(text, "Status") or ""
            question_id = path.name.split("-", 1)[0]
            statuses[question_id] = status
            resolution = section_lines(text, "## Resolution")
            resolution_text = "\n".join(resolution or []).strip()
            if status == "answered" and (not resolution_text or resolution_text.lower() == "unresolved."):
                self.add("error", "unresolved-answered-question", path, "answered question lacks a resolution")
        todo = self.notebook / "TODO.md"
        if not todo.is_file():
            return
        for number, line in enumerate(todo.read_text(encoding="utf-8").splitlines(), 1):
            if not re.match(r"^- \[ \]", line.strip()):
                continue
            match = re.search(r"\bsrc:\s*(q\d{3})\b", line)
            if not match:
                continue
            question_id = match.group(1)
            if question_id not in statuses:
                self.add("error", "missing-todo-source", todo, f"line {number} references missing {question_id}")
            elif statuses[question_id] in {"answered", "obsolete"}:
                self.add("warning", "stale-todo", todo, f"line {number} references {question_id} ({statuses[question_id]})")

    def check_duplicate_ids(self) -> None:
        for note_type, prefix in (("questions", "q"), ("decisions", "d")):
            directory = self.notebook / "notes" / note_type
            seen: dict[str, Path] = {}
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob(f"{prefix}[0-9][0-9][0-9]-*.md")):
                identifier = path.name.split("-", 1)[0]
                if identifier in seen:
                    self.add("error", "duplicate-note-id", path, f"duplicates {relative(seen[identifier], self.root)}")
                seen[identifier] = path

    def check_index(self) -> None:
        if not self.notebook.is_dir():
            return
        inspection = self.index.inspect()
        if inspection.state == "missing":
            self.add("warning", "missing-index", inspection.path, "generated retrieval index is missing")
        elif inspection.state == "malformed":
            self.add("error", "malformed-index", inspection.path, "generated retrieval index header is invalid")
        elif inspection.state == "stale":
            message = (
                "source digest is current, but generated contents differ"
                if inspection.actual_digest == inspection.expected_digest
                else f"expected source digest {inspection.expected_digest}; recorded {inspection.actual_digest}"
            )
            self.add(
                "warning",
                "stale-index",
                inspection.path,
                message,
            )

    def check_machine_json(self) -> None:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no", "--", "experiments"],
                cwd=self.root,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            return
        if result.returncode != 0:
            return
        for line in result.stdout.splitlines():
            path_text = line[3:].split(" -> ")[-1]
            if path_text.endswith(".json"):
                self.add("error", "modified-machine-json", self.root / path_text, f"git reports {line[:2].strip() or 'modified'}")

    def safe_fix(self) -> int:
        fixed = 0
        for path in self.study_files():
            text = path.read_text(encoding="utf-8")
            if section_lines(text, "## Durable notes") is None:
                path.write_text(text.rstrip() + "\n\n## Durable notes\n", encoding="utf-8")
                fixed += 1
        for path in [directory / "README.md" for directory in self.experiment_dirs()]:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            if section_lines(text, "## Durable notes") is None:
                path.write_text(text.rstrip() + "\n\n## Durable notes\n", encoding="utf-8")
                fixed += 1
        for path in self.session_files():
            text = path.read_text(encoding="utf-8")
            additions = []
            if section_lines(text, "## Things to extract later") is None:
                additions.append("## Things to extract later")
            if section_lines(text, "## Durable notes") is None:
                additions.append("## Durable notes")
            if additions:
                path.write_text(text.rstrip() + "\n\n" + "\n\n".join(additions) + "\n", encoding="utf-8")
                fixed += 1
        for directory in self.experiment_dirs():
            notes = directory / "notes.md"
            if not notes.is_file():
                continue
            text = notes.read_text(encoding="utf-8")
            additions = []
            if section_lines(text, "## Things to extract later") is None:
                additions.append("## Things to extract later")
            if additions:
                notes.write_text(text.rstrip() + "\n\n" + "\n\n".join(additions) + "\n", encoding="utf-8")
                fixed += 1
        return fixed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".", help="project root (default: current directory)")
    parser.add_argument("--notebook", default="lab-notebook", help="notebook path relative to project")
    parser.add_argument("--experiments", default="experiments/*", help="experiment directory glob")
    parser.add_argument("--sessions", help="sessions path relative to project")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    parser.add_argument("--strict", action="store_true", help="return nonzero for warnings")
    parser.add_argument("--fix", action="store_true", help="add only missing empty structural sections")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    doctor = Doctor(Path(args.project), args.notebook, args.experiments, args.sessions)
    fixed = doctor.safe_fix() if args.fix else 0
    issues = doctor.run()
    counts = {level: sum(item.level == level for item in issues) for level in ("error", "warning")}
    if args.as_json:
        print(json.dumps({"fixed": fixed, "summary": counts, "issues": [asdict(item) for item in issues]}, indent=2))
    else:
        if fixed:
            print(f"Fixed {fixed} structural file(s).")
        for item in issues:
            print(f"{item.level.upper():7} {item.code:28} {item.path}\n        {item.message}")
        print(f"Summary: {counts['error']} errors, {counts['warning']} warnings")
    if counts["error"] or (args.strict and counts["warning"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
