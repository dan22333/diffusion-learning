#!/usr/bin/env python3
"""Agent-facing relationship audit workflow for a research journal."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from explorer.feedback import FeedbackStore
from explorer.model import RELATION_TYPES, ResearchModel
from relationship_reviews import LedgerError, RelationshipReviews, pair_key


def project_paths(args: argparse.Namespace) -> tuple[Path, str]:
    return Path(args.project).resolve(), args.notebook


def status_command(args: argparse.Namespace) -> int:
    root, notebook = project_paths(args)
    reviews = RelationshipReviews(root, notebook)
    status = reviews.status()
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    notes, pairs = status["notes"], status["pairs"]
    print(f"Relationship review ledger: {status['ledger']}")
    print(f"Durable notes: {notes['current']}/{notes['total']} audited at their current content")
    print(f"Review debt: {notes['stale']} changed, {notes['unreviewed']} never audited")
    print(f"Pair decisions: {pairs['current']} current, {pairs['stale']} stale")
    if pairs["by_verdict"]:
        print("Current pair verdicts: " + ", ".join(f"{key}={value}" for key, value in pairs["by_verdict"].items()))
    if status["last_audit"]:
        print(f"Last note audit: {status['last_audit']['timestamp']} by {status['last_audit']['reviewer']}")
    return 0


def init_command(args: argparse.Namespace) -> int:
    root, notebook = project_paths(args)
    reviews = RelationshipReviews(root, notebook)
    changed = reviews.ensure()
    print(f"{'Created' if changed else 'Present'}: {reviews.path}")
    return 0


def packet_command(args: argparse.Namespace) -> int:
    root, notebook = project_paths(args)
    reviews = RelationshipReviews(root, notebook)
    snapshot = ResearchModel(root, notebook, args.experiments).build()
    status = reviews.status()
    documents = {document["path"]: document for document in snapshot["documents"]}
    candidates = snapshot["relationship_candidates"]["items"][: args.limit]
    packet = {
        "scope": {
            "stale_notes": status["notes"]["stale_paths"],
            "unreviewed_notes": status["notes"]["unreviewed_paths"],
        },
        "candidates": [
            {
                **candidate,
                "notes": [
                    {
                        "path": path,
                        "title": documents[path]["title"],
                        "status": documents[path]["status"],
                        "summary": documents[path]["summary"],
                        "confidence": documents[path]["confidence"],
                    }
                    for path in (candidate["a"], candidate["b"])
                ],
            }
            for candidate in candidates
        ],
        "warning": "Candidate evidence is deterministic retrieval support, not a semantic relationship assertion.",
    }
    if args.json:
        print(json.dumps(packet, indent=2, sort_keys=True))
        return 0
    print("# Relationship audit packet")
    print()
    print("This packet narrows reading work. An agent must read both canonical notes and their evidence before recording a verdict.")
    print()
    print(f"Changed notes: {len(packet['scope']['stale_notes'])}; unaudited notes: {len(packet['scope']['unreviewed_notes'])}; candidate comparisons: {len(candidates)}")
    for index, candidate in enumerate(packet["candidates"], 1):
        print(f"\n## {index}. {candidate['notes'][0]['title']} ↔ {candidate['notes'][1]['title']}")
        for note in candidate["notes"]:
            print(f"- `{note['path']}` [{note['status']}]: {note['summary'] or 'No indexed summary.'}")
        evidence = candidate["evidence"]
        for pattern in evidence.get("patterns", []):
            print(f"- Retrieval signal: {pattern}")
        if evidence.get("distinctive_terms"):
            print(f"- Shared terms: {', '.join(evidence['distinctive_terms'])}")
        if evidence.get("shared_sources"):
            print(f"- Shared sources: {', '.join(evidence['shared_sources'])}")
    return 0


def audit_command(args: argparse.Namespace) -> int:
    root, notebook = project_paths(args)
    reviews = RelationshipReviews(root, notebook)
    paths = reviews.note_paths() if args.all else [reviews.resolve_note(path) for path in args.paths]
    if not paths:
        raise LedgerError("provide durable note paths or use --all")
    events = reviews.audit_many(paths, args.reviewer, args.note)
    print(f"Recorded {len(events)} digest-bound note audit(s) in {reviews.path}")
    return 0


def relationship_payload(args: argparse.Namespace) -> dict[str, str] | None:
    values = (args.type, args.relationship_source, args.relationship_target, args.reason)
    if not any(values):
        return None
    if not all(values):
        raise LedgerError("typed/proposed reviews require --type, --relationship-source, --relationship-target, and --reason")
    if args.type not in RELATION_TYPES:
        raise LedgerError(f"unsupported relationship type: {args.type}")
    return {"type": args.type, "source": args.relationship_source, "target": args.relationship_target, "reason": args.reason}


def record_command(args: argparse.Namespace) -> int:
    root, notebook = project_paths(args)
    reviews = RelationshipReviews(root, notebook)
    relationship = relationship_payload(args)
    if args.verdict in {"typed", "proposed"} and not relationship:
        raise LedgerError(f"{args.verdict} verdict requires relationship details")
    event = reviews.record_pair(
        Path(args.left), Path(args.right), args.verdict, args.reviewer, args.note, relationship
    )
    print(f"Recorded {event['verdict']} review for {' :: '.join(event['paths'])}")
    return 0


def propose_command(args: argparse.Namespace) -> int:
    root, notebook = project_paths(args)
    reviews = RelationshipReviews(root, notebook)
    source = reviews.resolve_note(args.source)
    target = reviews.resolve_note(args.target)
    source_raw, target_raw = (
        source.relative_to(root).as_posix(),
        target.relative_to(root).as_posix(),
    )
    relationship = {"type": args.type, "source": source_raw, "target": target_raw, "reason": args.reason}
    pair = list(pair_key(source_raw, target_raw))
    reviews.record_pair(source, target, "proposed", args.reviewer, args.note, relationship)
    record = FeedbackStore(root).create(
        {
            "kind": "relation",
            "message": f"{source.name} {args.type} {target.name}",
            "requested_action": f"Verify the evidence, then add an explicit {args.type} relationship from {source_raw} to {target_raw}: {args.reason}",
            "target": {
                "path": source_raw,
                "heading_path": ["Relationships"],
                "text_quote": {"exact": "", "prefix": "", "suffix": ""},
                "snapshot_lines": [1, 1],
                "relation": {"pair": pair, "verdict": "proposed", **relationship},
            },
        }
    )
    print(f"Created relationship proposal {record['id']} and recorded its digest-bound review")
    return 0


def apply_command(args: argparse.Namespace) -> int:
    """Promote already-applied, explicitly approved outbox proposals into the ledger."""
    root, notebook = project_paths(args)
    reviews = RelationshipReviews(root, notebook)
    store = FeedbackStore(root)
    records = store.records()
    applied = 0
    model = ResearchModel(root, notebook, args.experiments).build()
    typed = {
        (edge["source"], edge["target"], edge["type"], edge.get("reason", ""))
        for edge in model["edges"]
        if edge["type"] in RELATION_TYPES
    }
    for proposal_id in args.proposal_ids:
        record = records.get(proposal_id)
        relation = ((record or {}).get("target") or {}).get("relation") or {}
        required = (relation.get("source"), relation.get("target"), relation.get("type"), relation.get("reason"))
        if not record or relation.get("verdict") != "proposed" or not all(required):
            raise LedgerError(f"{proposal_id}: not a complete relationship proposal")
        if tuple(required) not in typed:
            raise LedgerError(
                f"{proposal_id}: canonical relationship is not present yet; an agent must verify and edit the source note before apply"
            )
        reviews.record_pair(
            Path(relation["source"]),
            Path(relation["target"]),
            "typed",
            args.reviewer,
            f"Approved proposal {proposal_id}. {args.note}".strip(),
            {"source": required[0], "target": required[1], "type": required[2], "reason": required[3]},
        )
        store.update(proposal_id, "resolved", f"Applied and tracked by {args.reviewer}")
        applied += 1
    print(f"Promoted {applied} applied proposal(s) into the tracked review ledger")
    return 0


def bootstrap_command(args: argparse.Namespace) -> int:
    root, notebook = project_paths(args)
    reviews = RelationshipReviews(root, notebook)
    snapshot = ResearchModel(root, notebook, args.experiments).build()
    current = reviews.current_pair_reviews()
    added = 0
    for edge in snapshot["edges"]:
        if edge["type"] not in RELATION_TYPES:
            continue
        key = pair_key(edge["source"], edge["target"])
        existing = current.get(key)
        relationship = {key: edge[key] for key in ("source", "target", "type", "reason")}
        if existing and existing.get("verdict") == "typed" and existing.get("relationship") == relationship:
            continue
        reviews.record_pair(
            Path(edge["source"]), Path(edge["target"]), "typed", args.reviewer, args.note, relationship
        )
        added += 1
    print(f"Recorded {added} existing explicit relationship(s) as reviewed")
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=".")
    parser.add_argument("--notebook", default="lab-notebook")
    parser.add_argument("--experiments", default="experiments/*")
    parser.add_argument("--sessions")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create the tracked review ledger if absent")
    init.set_defaults(func=init_command)
    status = commands.add_parser("status", help="show digest-bound review coverage and debt")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=status_command)
    packet = commands.add_parser("packet", help="prepare deterministic evidence for an agent audit")
    packet.add_argument("--limit", type=int, default=20)
    packet.add_argument("--json", action="store_true")
    packet.set_defaults(func=packet_command)
    audit = commands.add_parser("audit", help="record notes that an agent actually reviewed")
    audit.add_argument("paths", nargs="*")
    audit.add_argument("--all", action="store_true")
    audit.add_argument("--reviewer", required=True)
    audit.add_argument("--note", default="")
    audit.set_defaults(func=audit_command)
    record = commands.add_parser("record", help="record a semantic pair verdict")
    record.add_argument("left")
    record.add_argument("right")
    record.add_argument("--verdict", choices=sorted({"typed", "proposed", "keep-related", "not-related", "deferred"}), required=True)
    record.add_argument("--reviewer", required=True)
    record.add_argument("--note", default="")
    record.add_argument("--type")
    record.add_argument("--relationship-source")
    record.add_argument("--relationship-target")
    record.add_argument("--reason")
    record.set_defaults(func=record_command)
    propose = commands.add_parser("propose", help="place an agent-vetted relationship proposal in the feedback outbox")
    propose.add_argument("source")
    propose.add_argument("target")
    propose.add_argument("--type", choices=sorted(RELATION_TYPES), required=True)
    propose.add_argument("--reason", required=True)
    propose.add_argument("--reviewer", required=True)
    propose.add_argument("--note", default="")
    propose.set_defaults(func=propose_command)
    apply = commands.add_parser("apply", help="track approved proposals after an agent has edited canonical notes")
    apply.add_argument("proposal_ids", nargs="+")
    apply.add_argument("--reviewer", required=True)
    apply.add_argument("--note", default="")
    apply.set_defaults(func=apply_command)
    bootstrap = commands.add_parser("bootstrap", help="record existing explicit relationships as reviewed")
    bootstrap.add_argument("--reviewer", required=True)
    bootstrap.add_argument("--note", default="Imported existing explicit relationships.")
    bootstrap.set_defaults(func=bootstrap_command)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        return args.func(args)
    except (LedgerError, OSError, ValueError) as error:
        print(f"relationships: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
