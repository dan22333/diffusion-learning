#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from explorer.feedback import FeedbackStore
    from explorer.model import ResearchModel
else:
    from .feedback import FeedbackStore
    from .model import ResearchModel


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Inspect the research-journal explorer feedback outbox")
    result.add_argument("--project", default=".")
    result.add_argument("--notebook", default="lab-notebook")
    result.add_argument("--experiments", default="experiments/*")
    result.add_argument("--state-dir", type=Path)
    subparsers = result.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--all", action="store_true")
    for command in ("show", "claim", "resolve", "dismiss", "act"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("id")
        if command in {"claim", "resolve", "dismiss"}:
            command_parser.add_argument("--note", default="")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = Path(args.project).resolve()
    model = ResearchModel(root, args.notebook, args.experiments)
    store = FeedbackStore(root, args.state_dir)
    snapshot = model.build()
    records = {record["id"]: record for record in store.list_resolved(snapshot)}
    if args.command == "list":
        values = records.values() if args.all else (record for record in records.values() if record["status"] in {"open", "claimed"})
        for record in values:
            target = record.get("target") or {}
            resolution = record.get("resolution") or {}
            line = resolution.get("line_start") or (target.get("snapshot_lines") or ["?"])[0]
            print(f"{record['id']}  {record['status']:9} {record['kind']:18} {target.get('path')}:{line}")
            print(f"  {record.get('message') or record.get('requested_action') or '(no message)'}")
        return 0
    record = records.get(args.id)
    if not record:
        print(f"feedback not found: {args.id}", file=sys.stderr)
        return 1
    if args.command == "show":
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0
    if args.command in {"claim", "resolve", "dismiss"}:
        status = {"claim": "claimed", "resolve": "resolved", "dismiss": "dismissed"}[args.command]
        print(json.dumps(store.update(args.id, status, args.note), indent=2, ensure_ascii=False))
        return 0
    target = record.get("target") or {}
    resolution = record.get("resolution") or {}
    line = resolution.get("line_start") or (target.get("snapshot_lines") or [1])[0]
    print(f"Research-journal feedback {record['id']}")
    print(f"Target: {target.get('path')}:{line}")
    print(f"Kind: {record.get('kind')}")
    print(f"Attachment: {resolution.get('attachment')}")
    if record.get("message"):
        print(f"Feedback: {record['message']}")
    if record.get("requested_action"):
        print(f"Requested action: {record['requested_action']}")
    quote = (target.get("text_quote") or {}).get("exact")
    if quote:
        print(f"Selected text: {quote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
