# Development Workflow

## Claude Code and Compound Engineering

This repository is maintained with Claude Code and the Compound Engineering Plugin.

Compound Engineering is used for structured engineering work in this project.

Use it for:
- architecture changes
- ML model pipeline changes
- dataset sourcing, cleaning, validation, and training
- Telegram bot routing changes
- GPT fallback logic
- deployment and service configuration
- dependency updates
- risky refactoring
- pre-commit audits
- final technical summaries

Avoid using it for:
- typo fixes
- minor comments
- simple one-line config changes
- small local explanations
- trivial one-file fixes where risk is low

## Preferred Workflow

1. `/ce-code-review` — read-only audit before risky work.
2. `/ce-brainstorm` — explore possible solutions.
3. `/ce-plan` — prepare an implementation plan.
4. `/ce-work` — execute the plan.
5. `/ce-code-review` — validate the result.
6. `/ce-compound` — preserve reusable lessons and project knowledge.

## Project-Specific Safety Rules

- Do not remove GPT fallback.
- Do not modify production launchd or service files unless explicitly required.
- Do not change bot routing unless the task specifically requires it.
- Do not commit datasets, model weights, logs, caches, run artifacts, or local secrets.
- Preserve existing dataset validation and training guards.
- Preserve stage separation:
  - Stage A: detector and structural filtering.
  - Stage B: species classifier.
- Always provide rollback notes after risky changes.
- Always produce a strict technical summary explaining:
  - what was changed
  - why it was changed
  - what was not changed
  - how it was validated
  - remaining risks
