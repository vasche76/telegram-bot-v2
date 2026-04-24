
## Compound Engineering Workflow

This project uses the Compound Engineering Plugin in Claude Code for complex engineering tasks.

Use Compound Engineering when the task involves:
- architecture changes
- ML pipeline changes
- dataset ingestion, validation, or training
- deployment, launchd, or service changes
- risky refactoring
- code review before commit
- final technical summaries

Preferred flow:
1. `/ce-code-review` for read-only audit
2. `/ce-brainstorm` for solution exploration
3. `/ce-plan` for implementation planning
4. `/ce-work` for execution
5. `/ce-code-review` for final validation
6. `/ce-compound` to preserve reusable lessons

Do not use the full CE workflow for trivial one-file changes unless risk is high.

Safety rules:
- Do not remove GPT fallback.
- Do not modify launchd or service files without explicit intent.
- Do not commit datasets, model weights, logs, caches, or run artifacts.
- Always provide rollback notes after risky changes.
- Always produce a strict technical summary.
