# Documentation Standards

**Core principle**: Concise, high-signal documentation that answers "what, why, how" in minimal words.

## Key Points
- **Tone**: Technical, direct, no fluff. Avoid marketing language
- **Structure**: Overview → Key points → Examples → Reference
- **Files**: Target <200 lines per file. Split large docs into focused sub-files
- **Format**: Markdown with tables for reference data, code blocks for examples
- **Architecture docs**: Include ASCII flow diagrams for pipeline/process docs

## Required Sections
- Every service doc: Purpose, dependencies, config, key behaviors
- Every guide: Prerequisites, steps, expected output
- Every concept: 1-3 sentence definition, 3-5 bullet key points, minimal example

## What NOT to Do
- No duplicate information across docs (cross-reference instead)
- No installation steps that belong in a README
- No screenshots (use ASCII diagrams)
