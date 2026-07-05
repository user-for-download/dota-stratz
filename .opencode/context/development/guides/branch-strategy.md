# Git Workflow

**Core principle**: Single-main trunk with feature branches. Linear history preferred.

## Branching
- `main` — Production-ready. Always deployable.
- `feat/{name}` — New features. Rebase onto main before merge.
- `fix/{name}` — Bug fixes. Fast-forward merge preferred.
- `chore/{name}` — Refactors, tooling, CI changes.

## Commits
- Conventional commits format: `type(scope): description`
- One logical change per commit (not "WIP" or "fixup")
- No force-push to shared branches (rebase only your own)

## Before Merge
1. `make check` passes (fmt + vet + test)
2. All new env vars in `deploy/.env.example`
3. Metrics added for new operations
4. Context files updated if architecture changed
