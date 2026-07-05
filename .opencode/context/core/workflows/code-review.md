# Code Review Workflow

**Core principle**: Catch bugs, enforce patterns, share knowledge. Every change gets reviewed.

## Key Points
- **Gate**: All PRs before merge. No bypass for "small" changes
- **Scope**: Logic correctness, error handling, concurrency safety, config completeness
- **Format**: Report as checklist of findings with severity labels: `BLOCKER` / `WARNING` / `INFO`

## Review Checklist
- [ ] `make check` passes (fmt + vet + test)
- [ ] No deadlocks or goroutine leaks (check `shutdown` channels, `ctx.Done()`)
- [ ] All env vars have defaults and are in `deploy/.env.example`
- [ ] Metrics added for new operations (counters + histograms where applicable)
- [ ] DLQ routing exists for new queues (no silent drops)
- [ ] `ON CONFLICT DO NOTHING` for all DB inserts (idempotency)
- [ ] New config fields documented in the relevant service's context

## Reporting Format
```
## Review: {file/feature}
**BLOCKER**: {critical issue with explanation}
**WARNING**: {potential issue, may not be triggered}
**INFO**: {suggestion, style, minor}
```
