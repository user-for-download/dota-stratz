# Extract — Create Context from Existing Documents

**Core concept**: Read a source file (code, docs, URL) and distill it into MVI-compliant context files.

## Workflow
1. Read source: `ARCHITECTURE.md`, `Makefile`, service code, etc.
2. Identify extractable knowledge: concepts, patterns, reference data
3. Split into focused context files (<200 lines each)
4. Use function-based structure: `concepts/`, `lookup/`, `guides/`
5. Update `navigation.md` with cross-references

## Splitting Rules
- One concept per file (e.g., `pipeline.md`, `services.md`, `database.md`)
- Reference data in `lookup/` (tables, ports, env vars)
- Step-by-step in `guides/` (how to run, how to deploy)
- Cross-reference instead of duplicating information
