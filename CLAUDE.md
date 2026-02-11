# Deep MCP — Project Instructions

## Autonomous Decision-Making Framework

### Risk Levels

**GREEN — autonomous (low risk).** Act without asking.
- Reversible actions (can undo)
- Local impact only
- Standard operations: reading files, formatting, refactoring without API changes, generating code within existing patterns

**YELLOW — inform (medium risk).** Act, then report decision with rationale.
- Noticeable project impact or side effects
- Adding/changing dependencies
- Architectural changes within a module
- Changing API contracts between internal modules

Format: state what was done, why, risks, how to revert.

**RED — escalate (high risk).** Always ask before acting.
- Irreversible actions
- Security-related changes
- External integrations, API keys, third-party calls
- Deleting critical data or files
- Push to remote, production deployment
- Changes to sandbox security boundaries (REPL restrictions, allowed builtins)

### Task Decomposition

Break tasks down when any threshold is exceeded:
- Estimated duration >30 min → split into 15-min chunks
- Touches >5 files → group by related files
- Spans >3 components → split by component
- Mixes risk levels → separate by risk level

A task is **atomic** when: ≤15 min, ≤3 files, single clear outcome, one risk level, no internal dependencies.

### Risk Aggregation

- Any HIGH risk subtask → escalate the entire plan
- All MEDIUM → inform + execute sequentially
- All LOW → autonomous execution
- Mixed LOW+MEDIUM → inform about the plan, execute

### Context Modifiers

**Elevate risk (+1 level):** production environment, real user data, external dependencies, financial/PII data.

**Lower risk (-1 level):** dev/staging environment, explicit user authorization, automated rollback available, sandboxed execution.

---

## Documentation Order

Follow PRD-centric sequence when creating project documentation:

1. Actors & user journeys
2. PRD (central document — vision, features, acceptance criteria)
3. Non-functional requirements
4. Architecture blueprint
5. Data workflows & API contracts
6. ADR (architecture decision records)
7. System requirements document
8. Traceability matrix

Existing docs in `docs/` (not committed): PRD.md, MVP Design.md, RLM-Research.md.

---

## Git Commits

Commitizen format. Small, atomic commits — one logical change per commit.

```
<type>(<scope>): <short description>

[optional body]
```

Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`, `ci`, `build`, `perf`, `style`

Examples:
- `feat(server): add tool registration endpoint`
- `fix(auth): validate token expiry before refresh`
- `refactor(core): extract config loader to separate module`

## Never Commit `docs/`

The `docs/` folder must NEVER be committed to git. Do NOT add it to `.gitignore` — just exclude it manually from every `git add` command. Never use `git add .` or `git add -A` — always add files explicitly by name.

## Security — Public Repository

This is a public repo. Before every commit, verify:

- No API keys, tokens, or secrets in code
- No `.env` files with real values (`.env.example` with placeholders is OK)
- No private URLs, credentials, or internal endpoints
- No hardcoded passwords or connection strings

If a secret is accidentally committed, immediately rotate it — git history is permanent.

## Lessons Learned

<!-- Add entries here when mistakes happen, so they don't repeat -->
