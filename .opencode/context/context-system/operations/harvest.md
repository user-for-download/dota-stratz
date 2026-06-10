# Harvest — Extract Knowledge from AI Summaries

**Core concept**: Convert conversation summary files (SESSION-*.md, CONTEXT-*.md, *OVERVIEW.md) into permanent context entries, then clean up.

## Workflow
1. Read target summary file
2. Identify key concepts, decisions, patterns, blockers
3. Create/update context files in appropriate category
4. Update `navigation.md` if new files added
5. Archive or delete the source summary

## Criteria
- **Decisions**: Record in the relevant domain concept file
- **Patterns**: Record in `development/concepts/go-patterns.md`
- **Blockers**: Record in `errors/` if recurring
- **Transient notes**: Delete (not worth preserving)
