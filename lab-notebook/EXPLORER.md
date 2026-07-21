# Research-journal explorer

## Contents

- Launch and trust boundary
- Live indexing and raw artifacts
- Feedback outbox
- Knowledge evolution and typed relationships
- Project-local synchronization

## Launch and trust boundary

Prefer the project-local runtime when present:

```bash
python3 <NOTEBOOK_DIR>/tools/explorer/server.py --project <PROJECT> --open
```

Fall back to the skill-bundled `scripts/explorer/server.py` when the notebook has
not been synchronized. The server prints a plain loopback URL and accepts only
`127.0.0.1`, `::1`, or `localhost`. Browser requests must use the server's exact
loopback host and port; state-changing browser requests must also come from that
loopback origin. There is no URL or API token. The Explorer therefore trusts
other processes running as the local user, while browser same-origin policy,
host validation, and loopback binding prevent ordinary remote or cross-origin
access. It never writes canonical notebook or experiment files. Enable
Finder/file-manager and editor launches only when the user requests them, using
`--allow-system-open` and an explicit `--editor-command` template.

The derived SQLite search/table cache and append-only feedback log live under
the repository's Git metadata (`git rev-parse --git-path research-journal`) or,
for a non-Git project, the user state directory. Use `--state-dir` for tests or
an explicit alternative.

The default state directory also owns a process lease, so launching the explorer
again for the same project reuses the existing server and prints its current URL
instead of opening another port. The server treats its live browser event stream
as an ownership signal. After the final browser disconnects, it waits five
minutes for refreshes or reconnections and then shuts itself down; set
`--idle-timeout <seconds>` to change the grace period or `--idle-timeout 0` to
opt out. Inspect or stop the project instance explicitly with:

```bash
python3 <NOTEBOOK_DIR>/tools/explorer/server.py --project <PROJECT> --status
python3 <NOTEBOOK_DIR>/tools/explorer/server.py --project <PROJECT> --stop
```

Shutdown removes connection metadata and releases the lease. A crash may
leave inert metadata behind, but the next launch safely replaces it after
acquiring the abandoned operating-system lock. When `--state-dir` is supplied,
pass the same value to launch, status, and stop; an alternate state directory is
an intentionally isolated instance.

## Live indexing and raw artifacts

The explorer watches human-authored notebook Markdown, including study cards,
and experiment artifacts,
debounces changes, rebuilds the derived graph/search index, and notifies the UI
over server-sent events. The debounce is trailing-edge: a rebuild starts only
after writes go quiet, with a bounded maximum wait so a live experiment that
streams files still refreshes periodically, and the filesystem poll may not
exceed a 50% duty cycle when scanning is slower than the configured interval.
The browser must preserve the current route, filters,
and graph focus across updates; the graph additionally preserves its viewport
and selection when a live refresh rebuilds the same scope. The snapshot sent to
the browser omits raw document text (blocks are the rendering source; feedback
reattachment runs server-side), JSON and static responses are gzip-compressed
when the client accepts it, and the event stream is one persistent connection
rather than one per change. When a refresh fails mid-session, the browser keeps
the last good snapshot visible behind a retry banner instead of replacing the
workspace with an error screen.

The overview is a deterministic research cockpit, not a file-recency feed. Read
current focus only from `ACTIVE.md`; derive the latest research outcome and
activity timeline from dated `JOURNAL.md` entries; derive open work from unchecked
`TODO.md` items; and derive knowledge health, contradictions, running experiments,
feedback, and relationship-review debt from their explicit canonical statuses.
Separate infrastructure entries from research activity by the linked session's
`kind`. Never rank heterogeneous documents as a generic "current state," infer
importance from path order, or treat filesystem modification time and `unknown`
dates as research recency. Empty and stale states must be stated honestly. This
dashboard requires no scheduler, LLM process, or canonical database beyond the
existing notebook workflow; it recomputes with the normal live snapshot.

Raw artifact access is read-only and restricted to the configured notebook and
experiment roots. Private `.extraction/` content is never exposed. TSV/CSV files
receive an on-demand SQLite table cache; JSON, JSONL/text, Markdown, images,
PDFs, sandboxed HTML, and office/binary downloads use type-specific viewers.
Do not add raw traces to the research full-text index by default.

Present artifacts as an experiment-aware workbench, not a flat file list. The
navigator hierarchy is experiment, relative directory, then file; groups remain
collapsed until opened or searched, and no arbitrary file is selected on entry.
Filters may narrow by experiment, viewer kind, and path. The detail header must
retain the repository path, size, modified time, experiment-card and notes links,
and any linked durable research. Table viewers expose row count, pages, sorting,
column visibility, filtering, and cell selectors. JSON viewers expose a searchable
tree and JSON Pointer selectors. Text viewers expose search, wrapping, line
numbers, and line-range selectors; Markdown additionally offers rendered and raw
modes. Make truncation explicit and incrementally loadable. On narrow screens,
show the navigator before selection and a viewer-first detail with an explicit
back action afterward; never force the user through the full hierarchy to reach
the selected file.

Rendered Markdown links must retain normal repository-relative source syntax
while behaving as explorer navigation. Resolve relative document links against
the source file and open them through history-aware document routes; honor
heading fragments by scrolling to the addressed block. Route links to indexed
raw artifacts into the artifact viewer. Open safe external HTTP(S)/mailto links
outside the explorer, and visibly disable unresolved or unsafe targets instead
of letting the browser reinterpret them as server paths.
Treat `[[note]]`, `[[note#heading]]`, and `[[note|label]]` as first-class
internal links. Prefer a same-directory note, then a unique notebook-wide
filename stem; visibly disable ambiguous or missing wiki targets.

Document rendering must preserve visible H2-H6 section headings; the document
view supplies the H1 title once. For compatibility with older notes, escape lone
unescaped tildes outside code before Markdown parsing so approximation symbols
remain literal while intentional `~~strikethrough~~` still renders normally.
Render a single source newline as a visible line break; notebook prose frequently
uses one-line metadata and compact logs where collapsing newlines loses structure.

Search has one primary entry point: the persistent top bar. Typing there opens
the paragraph-level results view; do not duplicate it as a sidebar destination.
The query is URL-addressable (`#/search/<query>`): entering search pushes one
history entry and further typing rewrites it in place, so a search can be
shared, bookmarked, and restored. `/` or Cmd/Ctrl+K focuses the search field
from anywhere; Escape leaves it. The `#/relationships` route is an accepted
alias for the knowledge-evolution view's canonical `#/precedence`.

The journal is a first-class timeline view (`#/journal`): every dated
intent/outcome entry, newest first, filterable by category and text, each
linking to its recorded study, experiment, or session. The snapshot carries the
full entry list separately from the dashboard's capped activity feed. Long
documents expose an outline rail built from their headings; outline clicks and
`#/doc/<path>#<heading>` deep links scroll to the addressed section, and
outline navigation rewrites the URL anchor in place so sections are shareable.

Recognized experiment analysis artifacts open in structured viewers by name
plus shape: the experiment `summary.json` (config-versus-metric table with
per-task drill-down), `eval-summary.json` (per-criterion pass-rate strip with
failing and unstable lists), `reliability-*.json` (per-criterion pass
probability by run), and `agreement-*.json` (raw/kappa tables with disagreement
evidence). Detection never guesses: a shape mismatch falls back to the generic
JSON tree, and a Structured/Raw toggle always exposes the underlying JSON.
Two comparable artifacts open side by side at `#/compare/<a>::<b>` — aligned
numeric leaf deltas ranked by magnitude for JSON, a bounded line diff for
text — reachable from any comparable artifact's Compare action, which offers
same-named files in other experiments first. Comparison of truncated previews
must say so explicitly.

The graph inspector can trace the shortest recorded connection between two
notes: arm the trace on a selected node, tap the target, and the chain of hops
is highlighted on the canvas and explained hop by hop (edge type, direction,
rationale) in the inspector. Tracing searches the same edge set the current
cross-reference toggle allows and states plainly when no recorded chain exists.

The graph opens on a bounded research focus: use the `ACTIVE.md` target when it
resolves to a document, otherwise choose the strongest connected current note.
The corpus overview must aggregate documents into semantic type groups and
collapse relationships into counted source-to-note, explicit precedence,
citation/related, or other flows. Selecting a group reveals its status mix,
plain-language source-to-destination flow summaries, and ranked member
documents; selecting a member enters a one-hop graph. Rendering every document
at once is an explicit advanced option, not the corpus default. Hide generic
`related`/`cites` edges and unlinked documents behind separate, counted toggles.
The cross-reference toggle must describe both semantic states: hidden means the
graph contains only research lineage and explicit epistemic relationships;
included means citations and general related-note links are also used for
discovery but do not imply support, contradiction, or precedence. Keep corpus,
one-hop, and two-hop scopes separate from node selection.

Keep labels and nodes approximately constant in screen space. Cull labels by
selection, search match, degree, status, and type, then suppress collisions;
zooming or narrowing scope reveals more labels. Use visible arrowheads and
distinct color/dash treatment for directed edge types. Encode inactive and
contradicted statuses, auto-fit after scope changes, keep layouts deterministic,
and provide clickable type/status/date filters plus graph-local search
highlighting. The inspector must distinguish hover from selection, expose edge
direction, and allow expanding the complete connection list.

## Feedback outbox

Explorer feedback is the service's only write surface. Records are append-only
JSONL events outside the worktree. Address Markdown selections with:

- project-relative path;
- heading path;
- exact selected text plus prefix/suffix context;
- snapshot line range; and
- block fingerprint.

Reattach after edits using exact text and context, then heading-scoped fuzzy
matching. Report `attached`, `moved`, or `orphaned`; never silently retarget an
ambiguous comment. Artifact selectors may use a JSON Pointer, table row/key and
columns, or text line range.

Use project-local `tools/explorer/feedback_cli.py`, falling back to the bundled
copy, for `list`, `show`, `claim`, `resolve`, `dismiss`, and `act`. `act` prints
the addressable request; the assistant must still follow the normal journal
authority, capture, extraction, and doctor workflows.

## Knowledge evolution and typed relationships

The explorer's Relationships navigation opens a Knowledge evolution view. Its
primary surface contains only explicit typed relationships recorded in notes.
Group `supersedes` and `refines` as evolution, surface unresolved `contradicts`
as attention items, and group `supports`, `complements`, and `answers` as
evidence and resolution. An empty authoritative view is valid and must state
that the explorer does not infer knowledge evolution.

Machine-generated comparisons are a secondary, hard-bounded review queue, not
research content. Require a non-lexical evidence gate such as an existing
generic cross-reference, a rare shared source, or a lifecycle pattern; lexical
similarity alone must never surface a candidate. Common provenance hubs must not
inflate ranking. Do not assign a relationship type or direction automatically.
Show observable evidence, omit raw scores, display no more than ten comparisons,
and suppress the section entirely when none qualify. Persist keep-related,
not-related, defer, and relationship-proposal events through the append-only
feedback side channel. Only reviewed note edits become authoritative.

The tracked `<NOTEBOOK_DIR>/relationship-reviews.jsonl` ledger complements the
private outbox. It records full-file digests for notes an agent has semantically
audited and for pair verdicts. A changed digest invalidates the prior coverage;
the explorer must show changed notes, never-audited notes, stale pair decisions,
open proposals, accepted-but-unapplied typed decisions, and the last agent audit.
This panel is accountability metadata, not an inference engine. The browser stays
read-only with respect to the ledger; an active coding agent promotes reviewed
outbox decisions through `tools/relationships.py`.

Use an optional `## Relationships` section on durable notes. Each item has one
of these directed types:

```md
- supersedes: [older finding](relative/path.md) — replacement rationale
- contradicts: [earlier claim](relative/path.md) — conflicting evidence
- refines: [broader claim](relative/path.md) — narrower scope
- complements: [related finding](relative/path.md) — distinct compatible result
- supports: [existing claim](relative/path.md) — additional evidence
- answers: [open question](relative/path.md) — resolution
```

Every relationship requires a short evidence-backed rationale. Dates alone
never establish precedence. Prefer explicit relationships, current status,
evidence directness, confidence, then freshness. `supersedes` targets must carry
superseded status; `answers` targets must be answered questions with a resolution;
open contradictions remain visible until adjudicated. Reject self-links,
duplicates, dangling targets, and cycles in supersedes/refines chains.

## Project-local synchronization

`enable` copies `scripts/doctor.py`, `scripts/index.py`,
`scripts/relationships.py`, `scripts/relationship_reviews.py`, and the complete
`scripts/explorer/` runtime into `<NOTEBOOK_DIR>/tools/`. `sync-tools` performs
the same operation for an existing notebook only after inspecting the worktree
and comparing project-local copies. Do not overwrite locally modified tools
without explicit confirmation. Regenerate `INDEX.md` and run project-local
`doctor.py --strict` after synchronization.

The canonical editable frontend lives in the skill's `ui-src/` directory; its
`package.json` and tests live at the skill root. Run `npm run build` at the
skill root to produce the vendored `scripts/explorer/static/app.js`. Project
notebooks intentionally receive the built, dependency-free runtime rather than
the frontend toolchain.

Keep the explorer usable inside embedded browser surfaces: the main workspace,
not the outer document, owns vertical scrolling and exposes a stable scrollbar.
Bundle interface fonts locally, retain visible keyboard focus, and treat spacing,
type hierarchy, control states, and responsive behavior as part of the runtime
contract rather than optional decoration. Feedback and proposal dialogs are
accessible dialogs: dialog role, Escape-to-close, initial focus, and focus
return, and every write path surfaces its failure in place instead of hanging.
Icon-only navigation retains accessible names, and the live-status indicator is
a polite live region. View filters (graph toggles, artifact experiment/format,
activity scope) persist per served origin and fall back to safe defaults when a
persisted value no longer exists.
