# Deep MCP — Project Instructions

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
