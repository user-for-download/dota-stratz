# Organize — Restructure Flat Context Files

**Core concept**: Move flat context files into the function-based directory structure.

## Workflow
1. Scan target directory for unstructured `.md` files
2. Classify each file:
   - `concepts/` — What something is (1-3 sentence definition, key points)
   - `examples/` — Working code snippets (<10 lines per example)
   - `guides/` — Step-by-step workflows
   - `lookup/` — Quick reference tables
   - `errors/` — Recurring issues and solutions
3. Move/rename files into the correct subdirectory
4. Update `navigation.md`
5. Validate against MVI standards (<200 lines, scannable)

## Dry Run
```bash
/context organize development/ --dry-run
# Shows what would move where without making changes
```
