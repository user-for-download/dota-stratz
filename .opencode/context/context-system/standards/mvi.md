# Minimal Viable Information (MVI) Principle

**Core concept**: Every context file must be scannable in <30 seconds. Enforce by keeping files <200 lines and content to the essential minimum.

## Rules
- **File size**: <200 lines absolute. Split if exceeded
- **Core concept**: 1-3 sentences answering "what is this?"
- **Key points**: 3-5 bullets of the most important takeaways
- **Minimal example**: <10 lines of code or config
- **Reference link**: Point to full documentation or source file

## Why
- Context files are loaded by AI agents on every task
- Long files waste tokens and slow down response
- Short files ensure the AI actually reads the relevant context

## Enforcement
- `navigation.md` links should point to existing files
- Every new context file must pass the "30-second scan" test
- Use cross-references instead of duplicating content across files
