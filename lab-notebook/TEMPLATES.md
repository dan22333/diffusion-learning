# Research-journal templates

Use these controlled vocabularies exactly:

- Artifact `kind`: `study | experiment | session | reading | infrastructure | synthesis`
- Artifact `status`: `planned | running | done | partial | abandoned | superseded`
- Learning `Status`: `tentative | supported | contradicted | superseded`
- Question `Status`: `open | in-progress | answered | blocked | obsolete`
- Decision `Status`: `active | superseded`

Naming conventions:

| Type | Pattern | Example |
|---|---|---|
| Study | short stable slug | `judge-quality-v2.md` |
| Existing experiment | adopt the tool's directory convention | `20260629-153000` |
| Notebook session | `YYYY-MM-DD-topic.md` | `2026-06-29-parser-investigation.md` |
| Learning | `YYYY-MM-DD-short-claim.md` | `2026-06-29-normalization-hides-errors.md` |
| Question | `qNNN-short-question.md` | `q003-separate-validity-correctness.md` |
| Decision | `dNNN-short-decision.md` | `d002-track-validity-separately.md` |
| Meeting | `YYYY-MM-DD-topic.md` | `2026-06-29-stakeholder-feedback.md` |

Allocate question/decision numbers in one writer: scan the maximum existing ID,
reserve a consecutive batch, then create files.

In prose, escape an approximation tilde as `\~` (for example, `\~5%`). Use
paired `~~text~~` only for intentional strikethrough.

## Root activation stanza

Resolve root instruction-file symlinks and list only the physical file targets in
`instruction_files`.

```md
<!-- RESEARCH-MODE:ACTIVE notebook=<NOTEBOOK_DIR> studies=<NOTEBOOK_DIR>/studies experiments=<EXPERIMENTS_GLOB> sessions=<NOTEBOOK_DIR>/sessions instruction_files=<FILES> -->

## Research mode (ACTIVE)

This project keeps a self-contained research notebook under `<NOTEBOOK_DIR>/`
(protocol: `<NOTEBOOK_DIR>/AGENTS.md`; templates:
`<NOTEBOOK_DIR>/TEMPLATES.md`; automation: `<NOTEBOOK_DIR>/tools/` or the
`research-journal` skill).

- Orient from `<NOTEBOOK_DIR>/ACTIVE.md`, generated `INDEX.md`, recent
  `JOURNAL.md` entries, and relevant canonical notes. Check the index first;
  regenerate when writes are authorized or use `--stdout` when read-only.
- Keep multi-experiment directions in `<NOTEBOOK_DIR>/studies/`; capture
  experiment observations in each child experiment's `notes.md`; route other
  research to `<NOTEBOOK_DIR>/sessions/`. Never invent a pseudo-experiment.
- At the end of substantial work, append a Memory Delta to the target and a short
  linked entry to `JOURNAL.md`.
- Extract durable notes with two-way links and mark queue items `[x]` with their
  destination.
- Keep experiment cards current; never edit machine-generated artifacts.
- Explicit read-only/no-modification instructions override notebook writes.
- Run `python3 <NOTEBOOK_DIR>/tools/doctor.py --strict` to check integrity; use
  `python3 <NOTEBOOK_DIR>/tools/explorer/server.py --project . --open` for the
  live read-only explorer.
- After durable-note changes, use `tools/relationships.py status` and have an
  active agent audit changed notes; deterministic tools only report review debt.

To turn this off, invoke `research-journal exit` (`$research-journal exit` in
Codex or `/research-journal exit` in Claude Code); remove this block from every
listed instruction target but retain the notebook.

<!-- /RESEARCH-MODE -->
```

## ACTIVE.md

```md
# Active research target

Target: none
Updated: YYYY-MM-DD

Set `Target` to a project-relative study file, experiment directory, or session
file. Do not infer a target from the newest timestamp when this value is `none`.
```

## JOURNAL.md entry

```md
## YYYY-MM-DD — Short session title

Intent: One sentence.
Outcome: One or two sentences, including uncertainty or blockers.
Record: [study, experiment, or session](relative/path).
```

## Study card

```md
---
kind: study
status: planned
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# Study title

## Objective

What overarching direction coordinates the child experiments?

## Research questions

## Design commitments

Shared constraints, sampling rules, metrics, and gates that child experiments
must inherit.

## Experiment registry

| Stage | Status | Experiment | Purpose |
|---|---|---|---|
| 0 | planned | _Not created_ | ... |

Every created child experiment links back under `## Parent study`; replace the
placeholder with a relative link and keep the registry current.

## Open decisions

## Decision log

### YYYY-MM-DD

- Decision and rationale.

## Assistant session memory deltas

## Durable notes
```

Study cards are human-authored current-state maps, not pseudo-experiments. Keep
machine outputs in real experiment directories, and keep detailed running
observations in each child's `notes.md` or a non-experiment session.

## Experiment README.md

```md
---
kind: experiment
status: planned
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# Experiment title

## Parent study

Optional. If this experiment belongs to a study, link its study card here and
add the experiment to that card's registry.

## Question

What are we trying to learn?

## Why this matters

## Setup

- Code version: `<git commit>`
- Dataset: `<version/path>`
- Model/config: `<model or config>`
- Command: `command.sh`
- Config: `config.yaml`

## What changed from previous experiment

## Results

## Interpretation

## Caveats

## Follow-ups

- [ ] ...

## Closeout log

### YYYY-MM-DD

Outcome and reason for the final status.

## Durable notes

- Learning: [title](../../<NOTEBOOK_DIR>/notes/learnings/file.md)
- Question: [title](../../<NOTEBOOK_DIR>/notes/questions/qNNN-file.md)
- Decision: [title](../../<NOTEBOOK_DIR>/notes/decisions/dNNN-file.md)
```

Update frontmatter and current summary sections surgically. Append dated closeout
log entries. Never edit sibling machine-generated JSON or traces.

## Experiment notes.md

```md
# Notes: experiment-name

## Running log

### YYYY-MM-DD HH:MM

- Observation:
- Possible interpretation:
- Need to check:

## Things to extract later

- [ ] Learning: candidate claim
- [x] Decision: selected approach → extracted: [dNNN](../../<NOTEBOOK_DIR>/notes/decisions/dNNN-file.md)
- [x] Question: obsolete lead → resolved: disproved by [result](README.md#results)

## Assistant session memory deltas

<!-- Append dated full or short deltas here. -->
```

Only unchecked items are pending. Every checked item must include `extracted:` or
`resolved:` and a destination or reason.

## Non-experiment session

```md
---
kind: reading
status: running
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# Session title

## Intent

## Running notes

## Things to extract later

- [ ] Learning: ...

## Assistant session memory deltas

## Durable notes

- Learning: [title](../notes/learnings/YYYY-MM-DD-file.md)
```

Use `kind: session`, `reading`, `infrastructure`, or `synthesis`. Use the same
controlled artifact statuses as experiment cards.

## Full Memory Delta

```md
### YYYY-MM-DD — session-id-or-topic

#### New durable learnings
- ...
#### New open questions
- ...
#### Decisions made
- ...
#### Experiments or sessions created/modified
- ...
#### Files changed
- ...
#### Follow-up actions
- [ ] ...
#### Tentative claims needing verification
- ...
```

## Short Memory Delta

```md
### YYYY-MM-DD — session-id-or-topic (short)

- Outcome: ...
- Durable change: none | ...
- Follow-up: none | ...
```

## Learning note

```md
# Short durable learning title
Date: YYYY-MM-DD
Status: tentative
Source:
- [source title](../../../experiments/<exp>/README.md)

## Claim

One durable claim.

## Evidence

## Why it matters

## Confidence

Low / Medium / High

## Caveats

## Follow-up

## Related

## Relationships

- supports: [related finding](relative/path.md) — evidence-backed rationale
```

## Question note

```md
# qNNN: Short question title
Status: open
Created: YYYY-MM-DD
Source:
- [source title](../../../experiments/<exp>/README.md)

## Question

## Why it matters

## Current evidence

## Next checks

- [ ] ...

## Resolution

Unresolved.

## Related

## Relationships

- refines: [earlier question](relative/path.md) — evidence-backed rationale
```

When answering the question, set `Status: answered`, replace `Unresolved.` with a
dated resolution and evidence link, backlink the answering learning/decision, and
complete or remove its TODO.

## Decision note

```md
# dNNN: Short decision title
Date: YYYY-MM-DD
Status: active
Source:
- [source title](../../../experiments/<exp>/README.md)

## Decision

## Reason

## Consequences

## Related

## Relationships

- supersedes: [older decision](relative/path.md) — replacement rationale
```

## Meeting note

```md
# Meeting topic — YYYY-MM-DD

## Context
## Key feedback
## Assumptions corrected
## New constraints
## Decisions
## New questions
## Action items
- [ ] ...
## Links
- Related experiment / decision / question: ...
```
