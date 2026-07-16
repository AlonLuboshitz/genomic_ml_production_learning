# CI/CD — GitHub Actions

<!--
  What this file covers:
  - What the CI pipeline does
  - When it triggers
  - The two jobs: lint and test
  - How to view results
-->

## What it does

Every push or pull request to `main` triggers an automated pipeline
that lints the code and runs all tests inside the Docker image.

This ensures broken code never merges to main.

## Workflow file

`.github/workflows/ci.yml` defines two parallel jobs:

### 1. `lint` (runs on the GitHub runner, no Docker)

- Checks out the code
- Sets up Python 3.12
- Installs dependencies via `pip install -e ".[dev]"`
- Runs `ruff check` for syntax/import errors
- Runs `ruff format --check` for consistent formatting

### 2. `test` (runs inside Docker)

- Builds the Docker image
- Runs `pytest tests/ -v` inside the container

## How to view results

1. Push or open a PR on GitHub
2. Go to **Actions** tab in the repo
3. Click the running/completed workflow
4. Expand a job to see individual steps

## Running the same checks locally

```bash
make lint       # ruff check + ruff format (dry-run)
make test       # pytest tests/ -v
make docker-build && docker run --rm genomics-ml-api pytest tests/ -v
```
