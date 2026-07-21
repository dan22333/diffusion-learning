# Research memory protocol for coding assistants

This project uses Markdown as a lightweight, shared lab notebook. **Capture
first, extract second, synthesize only when needed.** The project-local notebook
is canonical; it must remain usable without a personal/global skill or private
assistant memory. See `TEMPLATES.md` for controlled statuses and file skeletons,
and `EXPLORER.md` for the live explorer and feedback contract.

## Safety first

- Explicit read-only/no-modification instructions override notebook capture.
- Preserve unrelated worktree changes. Never modify machine-generated experiment
  JSON, traces, metrics, or outputs.
- Do not persist credentials, PII, confidential source material, hidden reasoning,
  or private transcript content. Transcript mining and delegation require explicit
  user and repository permission.
- Do not commit unless the user authorizes it.

## Files and routing

- `<NOTEBOOK_DIR>/JOURNAL.md` — short chronological intent/outcome index.
- `<NOTEBOOK_DIR>/ACTIVE.md` — explicit current target; `Target: none` means ask.
- `<NOTEBOOK_DIR>/INDEX.md` — generated retrieval map; never hand-edit it.
- `<NOTEBOOK_DIR>/relationship-reviews.jsonl` — tracked, digest-bound record of
  agent semantic audits; changing a note makes its prior audit stale.
- `<NOTEBOOK_DIR>/tools/{doctor.py,index.py,relationships.py,relationship_reviews.py,explorer/}` — project-local checks and
  the live read-only explorer.
- `<NOTEBOOK_DIR>/studies/` — human-authored cards for overarching research
  directions that coordinate multiple experiments.
- `<NOTEBOOK_DIR>/sessions/` — reading, infrastructure, synthesis, and other
  non-experiment research.
- `<EXPERIMENTS_GLOB>/README.md` — current human-authored experiment card.
- `<EXPERIMENTS_GLOB>/notes.md` — messy experiment log and extraction queue.
- `<NOTEBOOK_DIR>/notes/learnings/` — one durable claim per file.
- `<NOTEBOOK_DIR>/notes/questions/` — open research questions.
- `<NOTEBOOK_DIR>/notes/decisions/` — decisions affecting later work.
- `<NOTEBOOK_DIR>/notes/meetings/` — stakeholder notes.
- `<NOTEBOOK_DIR>/TODO.md` — actions only, each with a source.

Put actual experiments only in the existing experiment tree; never rename a
tool-owned directory. Put a multi-experiment direction in one study card and
link every child experiment in both directions. Study cards are current-state
maps, not containers for machine outputs or replacements for child cards and
notes. Put other non-experiment work in `sessions/`. `capture` and `delta`
require an explicit target or the path stored in `ACTIVE.md`; never guess from
the newest timestamp.

## During work

- Put messy observations in the target's notes; keep experiment cards clean.
- Add potential durable items as unchecked tasks under
  `## Things to extract later`.
- Do not bury a durable conclusion only in chat or private memory.
- Reconcile existing notebook notes before creating duplicates. When optional
  agent memory overlaps, make the project note canonical and cross-link it.

## End of session

A substantial session changes research code/artifacts, creates or revises a
claim or decision, materially advances an experiment, or spans multiple
investigative steps. Append a full Memory Delta to the target. A small session may
use the short form; omit empty boilerplate.

Every substantial delta or closeout must also append a 2-3 line linked entry to
`JOURNAL.md` containing intent and outcome. Update or clear `ACTIVE.md`.

## Extraction and closure

1. Read the target notes/card and recent journal context.
2. Create or update atomic learning, question, and decision notes.
3. Use relative Markdown source links and add reverse links under the source's
   `## Durable notes` section.
4. Mark every processed queue item `[x]` with
   `extracted: <relative-link>` or `resolved: <reason/link>`. Only `[ ]` is pending.
5. If a learning or decision answers a question, update that question to
   `Status: answered`, replace `Unresolved.`, cross-link it, and complete/remove
   its TODO.
6. Run `tools/relationships.py packet`, read changed durable notes and evidence,
   and record agent verdicts. Positive typed edits go through the feedback outbox
   unless the user already explicitly approved them.

Allocate `qNNN` and `dNNN` in a single-writer stage: scan the current maximum,
reserve the consecutive IDs required for the batch, then create them. Never let
parallel workers mint shared IDs or edit shared indexes.

## Editing model

- Machine outputs and chronological logs are immutable/append-only.
- Human-authored current-state metadata, summaries, question resolutions, TODOs,
  and backlinks may be surgically updated.
- Update an experiment card's metadata and current summary at closeout, then
  append a dated closeout-log entry. Do not append duplicate stale sections.
- Preserve uncertainty using status, evidence, confidence, caveats, and explicit
  supersession.
- Write approximation symbols as `\~` so Markdown renderers do not pair distant
  tildes into strikethrough. Use `~~text~~` only when deletion styling is intended.
- Express knowledge evolution with evidence-backed typed `## Relationships`
  links (`supersedes`, `contradicts`, `refines`, `complements`, `supports`, or
  `answers`). Freshness alone never establishes precedence. Treat automated
comparisons as untyped review work routed through the feedback side channel,
never as research claims.

The relationship ledger records what an agent actually reviewed and binds each
decision to full-file content digests. `doctor.py` and `index.py` may report stale
or missing coverage, but deterministic tools must never invent a relationship.
Use `tools/relationships.py status` for debt, `packet` for evidence preparation,
and the agent-owned audit/propose/apply flow for semantic changes.

## Retrieval

From the project root, run:

```bash
python3 <NOTEBOOK_DIR>/tools/index.py --check
```

If fresh, use `INDEX.md` as the map into canonical sources. If stale or missing
and writes are authorized, run `--write`. Under explicit read-only instructions,
run `--stdout` to compute the current map without touching disk and ignore the
stale tracked index.

Search relevant active/supported learnings, open questions, active decisions,
meetings, study cards, experiment cards, sessions, scratchpads, then raw outputs. Within each
layer prefer current status, freshness, direct evidence, and explicit
supersession. Surface contradictions and tentative claims instead of silently
choosing one. The index is a cache, never the source of truth.

After any authorized notebook write, regenerate the index with `--write`. This
includes capture, delta, closeout, extraction, cleanup, and retrofit workflows.

Launch the live explorer with project-local `tools/explorer/server.py`. It may
write only its derived cache and feedback event log outside the worktree; it
must never edit canonical notes or experiment artifacts. Use its typed graph,
knowledge-evolution relationships view, raw-artifact viewers, and addressable feedback outbox without
flattening uncertainty. See the global skill's `references/explorer.md` when it
is available. Repeated launches reuse one project server, which shuts down five
minutes after its last browser disconnects by default. Use the server's
`--status` and `--stop` controls for explicit lifecycle management. Its overview
is populated directly from `ACTIVE.md`, dated journal outcomes, unchecked TODOs,
explicit statuses, feedback, and relationship-review state; never maintain a
separate dashboard database or use file modification times as research recency.

## Status and cleanup

Run `python3 <NOTEBOOK_DIR>/tools/doctor.py --strict` in read-only mode. It checks
activation, the active pointer, journal freshness, controlled card metadata,
extraction markers, backlinks, question/TODO closure, duplicate IDs, generated
index freshness, relationship-review coverage, and modified machine JSON. `--strict` treats warnings as
failure; `--json` emits structured output. `--fix` may add only missing empty
structural sections. Regenerate the index separately after an authorized fix.

Cleanup uses small, reviewable patches. Never infer that an item was extracted,
rewrite historical logs, or manufacture research conclusions.

## Git guidance

Keep notebook changes with the research work that produced them. At experiment
closeout, ensure the card, scratchpad, journal entry, durable notes, TODO, and code
tell one coherent story. Stage or commit only when the user requests that
workflow, and never absorb unrelated worktree changes.
