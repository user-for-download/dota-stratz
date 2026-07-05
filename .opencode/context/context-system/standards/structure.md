# Function-Based Directory Structure

**Core concept**: Organize context files by function (what they do), not by source (where they came from).

## Standard Layout
```
{category}/
├── navigation.md       # File index + brief descriptions
├── concepts/           # "What is it?" — definitions, architectures
├── examples/           # "How does it work?" — code snippets
├── guides/             # "How do I do X?" — step-by-step
├── lookup/             # "What's the value?" — reference tables
└── errors/             # "Why did this break?" — recurring issues
```

## Categories
- **core/** — AI agent standards (code quality, docs, tests, workflows)
- **domain/** — Project-specific knowledge (pipeline, services, database)
- **deployment/** — Docker Compose, monitoring, env vars
- **development/** — Go patterns, git workflow, dev commands
- **context-system/** — How to manage context files themselves

## Rules
- No flat files directly in a category directory (except `navigation.md`)
- No file >200 lines (split; keep focused)
- `navigation.md` is the entry point for each category
- Links are relative from the context root
