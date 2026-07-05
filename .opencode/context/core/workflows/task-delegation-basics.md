# Task Delegation Basics

**Core principle**: Delegate complex/multi-file tasks to specialists. Handle simple edits directly.

## When to Delegate
- **4+ files** or >60min estimated → TaskManager for breakdown
- **Specialist work** (tests, review, docs) → route to TestEngineer / CodeReviewer / DocWriter
- **Multi-step dependencies** → TaskManager for parallel batches
- **Single file / simple fix** → handle directly (no delegation overhead)

## Subagent Routing
| Task Type | Subagent | When |
|-----------|----------|------|
| Complex feature | TaskManager | 4+ files, multi-step deps |
| Testing | TestEngineer | New module, bug fix |
| Code review | CodeReviewer | Before merge |
| Documentation | DocWriter | New feature, API change |
| DevOps | OpenDevopsSpecialist | CI/CD, Docker, infra |
| Frontend | OpenFrontendSpecialist | UI changes |
| Context | ContextOrganizer | Harvest, extract, organize |
| External docs | ExternalScout | Library API lookup |

## Context Bundle
When delegating, create `.tmp/context/{session-id}/bundle.md` with:
- Task description and objectives
- All loaded context files (standards + domain)
- Constraints and output format
- Subagent instructions: "Load context from bundle before starting"
