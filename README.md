# llmzoomcamp

[![CI](https://github.com/betsyhcamp/llmzoomcamp/actions/workflows/ci.yml/badge.svg)](https://github.com/betsyhcamp/llmzoomcamp/actions/workflows/ci.yml) ![Python 3.14](https://img.shields.io/badge/python-3.14-blue) [![License: Unlicense](https://img.shields.io/badge/license-Unlicense-cyan.svg)](LICENSE)

This repository contains my code and notes working through the 2026 cohort of LLM Zoomcamp. This is a 10 week hands-on course focusing on RAG, vector search, embeddings, agents, evaluation, monitoring, and orchestration.

The course repo from the instructors is here: https://github.com/DataTalksClub/llm-zoomcamp

______________________________________________________________________

## Tooling Requirements

- Python 3.14
- [uv](https://github.com/astral-sh/uv)
- [Task](https://taskfile.dev/)

Make sure these three tools are installed.

## Getting started

Install dependencies:

```bash
uv sync
```

Pre-commit hooks can be installed with:

```bash
pre-commit install
```

## Project structure
 ** **`WIP`** **
```text
.
├── 01-agentic-rag
│   ├──lessons 
│   └──code
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── .vscode
│   └── settings.json
├── .github
│   └── workflows
│       └── ci.yml
├── pyproject.toml
├── README.md
├── .env **(not committed)**
└── Taskfile.yml

```
- Tooling configuration lives in `pyproject.toml`
- Automation is centralized in `Taskfile.yml`


## How checks are organized

This project uses a **hybrid approach** for code quality:

- **Pre-commit hooks** handle file utilities (whitespace, YAML validation, secrets detection) and delegate linting/formatting to Taskfile
- **Taskfile** is the single source of truth for all project-specific checks (lint, format, md-check, test)
- **CI** runs individual tasks (`task pre-commit`, `task test`, etc.) for better visibility in GitHub Actions

`task pre-commit` handles file utilities plus lint, format-check, and md-check via delegation, so CI does not need separate steps for these.

## Notes

- Tool versions in `pyproject.toml` are intentionally unpinned; `uv.lock` is the reproducibility mechanism and should be committed.
- CI mirrors local commands exactly via the Taskfile.

