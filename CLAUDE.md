## Project Memory
`MEMORY.md` at repo root (NOT the stub in `~/.claude/projects/*/memory/`) — gitignored and local-only, so a fresh clone won't have one; create it on first learning worth keeping. Read session-start if present; update on new learning. Long evidence lives in `memory/*.md`, loaded on demand. Code is ground truth; MEMORY is lossy cache — trust code on conflict.

## Python Path
Whichever environment `pip install -e .` was run in (see README's Install section) — no fixed path, it's machine-specific.


## Process
1. Analyze problem; read relevant files.
2. **Plan first.** Surface assumptions, name confusion, present alternatives — don't pick silently. Good plan:
   * **Why before what**: motivation + rejected alternatives per decision
   * **Equations/pseudocode**: only if replacing ambiguous prose
   * **Explicit assumptions**: inherited / fixed / out of scope
   * **Design invariants**: properties that break silently if violated
   * **Dependency chain**: what must exist before what
   * **Scope boundary**: what is NOT done and why
   * When user says "update plan" or names a `@plan` file, read/edit matching local file under `plan/` in repo; do not create separate plan docs elsewhere.
3. Execute; check off tasks.
4. Verify working. Updates scale with scope: trivial fix, none; local change, docstring; architectural change, docstring and plan file; novel gotcha, dead-end, or safety issue, all three plus MEMORY (per its append rules).
5. High-level change summary per task.

## Goal-Driven Execution
Transform tasks into verifiable goals; verify by whatever means fits the change. Loop until verified. Multi-step: state brief plan with verify-step per item. Strong success criteria enable independent loop; weak ones force clarification.

## Coding Guidelines
* Minimum code that solves problem. Nothing speculative.
* No features beyond ask. No abstractions for single-use. No unrequested flexibility/configurability.
* Skip defensive edge-cases for impossible scenarios.
* Single readable linear flow; extract functions only for reuse or distinct concept.
* Clean structure, consistent naming.
* Surgical edits: touch only what task requires. Don't improve adjacent code/comments/formatting. Match existing style even if you'd differ. Mention unrelated dead code; don't delete it.
* Remove orphans YOUR changes created (imports/vars/fns). Leave pre-existing dead code alone unless asked.
* Replace magic strings/numbers with named constants when reused.
* Single-responsibility functions; catch specific exceptions, never bare `except`.
* Docstrings: purpose, I/O, assumptions, design decisions + rationale (what/why, alternatives rejected) — primary catch-up reference.
* Inline comments where logic non-obvious.
* If 200 lines could be 50, rewrite. Senior-engineer test: would they call this overcomplicated?



# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## RTK Commands by Workflow

### Build & Compile (80-90% savings)
```bash
rtk cargo build         # Cargo build output
rtk cargo check         # Cargo check output
rtk cargo clippy        # Clippy warnings grouped by file (80%)
rtk tsc                 # TypeScript errors grouped by file/code (83%)
rtk lint                # ESLint/Biome violations grouped (84%)
rtk prettier --check    # Files needing format only (70%)
rtk next build          # Next.js build with route metrics (87%)
```

### Test (60-99% savings)
```bash
rtk cargo test          # Cargo test failures only (90%)
rtk go test             # Go test failures only (90%)
rtk jest                # Jest failures only (99.5%)
rtk vitest              # Vitest failures only (99.5%)
rtk playwright test     # Playwright failures only (94%)
rtk pytest              # Python test failures only (90%)
rtk rake test           # Ruby test failures only (90%)
rtk rspec               # RSpec test failures only (60%)
rtk test <cmd>          # Generic test wrapper - failures only
```

### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git show            # Compact show (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push            # Ultra-compact confirmations
rtk git pull            # Ultra-compact confirmations
rtk git branch          # Compact branch list
rtk git fetch           # Compact fetch
rtk git stash           # Compact stash
rtk git worktree        # Compact worktree
```

Note: Git passthrough works for ALL subcommands, even those not explicitly listed.

### GitHub (26-87% savings)
```bash
rtk gh pr view <num>    # Compact PR view (87%)
rtk gh pr checks        # Compact PR checks (79%)
rtk gh run list         # Compact workflow runs (82%)
rtk gh issue list       # Compact issue list (80%)
rtk gh api              # Compact API responses (26%)
```

### JavaScript/TypeScript Tooling (70-90% savings)
```bash
rtk pnpm list           # Compact dependency tree (70%)
rtk pnpm outdated       # Compact outdated packages (80%)
rtk pnpm install        # Compact install output (90%)
rtk npm run <script>    # Compact npm script output
rtk npx <cmd>           # Compact npx command output
rtk prisma              # Prisma without ASCII art (88%)
rtk uv run <cmd>        # Compact uv project command output
```

### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%). Format flags (-c, -l, -L, -o, -Z) run raw.
rtk find <pattern>      # Find grouped by directory (70%)
```

### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk json <file>         # JSON structure without values
rtk deps                # Dependency overview
rtk env                 # Environment variables compact
rtk summary <cmd>       # Smart summary of command output
rtk diff                # Ultra-compact diffs
```

### Infrastructure (85% savings)
```bash
rtk docker ps           # Compact container list
rtk docker images       # Compact image list
rtk docker logs <c>     # Deduplicated logs
rtk kubectl get         # Compact resource list
rtk kubectl logs        # Deduplicated pod logs
```

### Network (65-70% savings)
```bash
rtk curl <url>          # Compact HTTP responses (70%)
rtk wget <url>          # Compact download output (65%)
```

### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk gain --history      # View command history with savings
rtk discover            # Analyze Claude Code sessions for missed RTK usage
rtk proxy <cmd>         # Run command without filtering (for debugging)
rtk init                # Add RTK instructions to CLAUDE.md
rtk init --global       # Add RTK to ~/.claude/CLAUDE.md
```

## Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->