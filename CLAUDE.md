<!-- RESEARCH-MODE:ACTIVE notebook=lab-notebook studies=lab-notebook/studies experiments=experiments/* sessions=lab-notebook/sessions instruction_files=CLAUDE.md -->

## Research mode (ACTIVE)

This project keeps a self-contained research notebook under `lab-notebook/`
(protocol: `lab-notebook/AGENTS.md`; templates:
`lab-notebook/TEMPLATES.md`; automation: `lab-notebook/tools/` or the
`research-journal` skill).

- Orient from `lab-notebook/ACTIVE.md`, generated `INDEX.md`, recent
  `JOURNAL.md` entries, and relevant canonical notes. Check the index first;
  regenerate when writes are authorized or use `--stdout` when read-only.
- Keep multi-experiment directions in `lab-notebook/studies/`; capture
  experiment observations in each child experiment's `notes.md`; route other
  research to `lab-notebook/sessions/`. Never invent a pseudo-experiment.
- At the end of substantial work, append a Memory Delta to the target and a short
  linked entry to `JOURNAL.md`.
- Extract durable notes with two-way links and mark queue items `[x]` with their
  destination.
- Keep experiment cards current; never edit machine-generated artifacts.
- Explicit read-only/no-modification instructions override notebook writes.
- Run `python3 lab-notebook/tools/doctor.py --strict` to check integrity; use
  `python3 lab-notebook/tools/explorer/server.py --project . --open` for the
  live read-only explorer.
- After durable-note changes, use `tools/relationships.py status` and have an
  active agent audit changed notes; deterministic tools only report review debt.

To turn this off, invoke `research-journal exit` (`$research-journal exit` in
Codex or `/research-journal exit` in Claude Code); remove this block from every
listed instruction target but retain the notebook.

<!-- /RESEARCH-MODE -->
