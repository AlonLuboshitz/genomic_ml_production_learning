# Docker — containerized API serving

<!--
  What this file covers:
  - What Docker is used for in this project
  - How to build, run, and stop the container
  - What each Docker-related file does
-->

## What Docker does for us

The FastAPI prediction service runs inside a Docker container so anyone
can run it without installing Python dependencies manually — just Docker.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Defines the image: Python 3.12-slim, installs the package, runs uvicorn on port 8000 |
| `.dockerignore` | Excludes cache, venv, .env, .git, models (except baseline) — keeps the image small |
| `compose.yaml` | Orchestrates a single `api` service — builds the image, maps ports, mounts volumes |

## Commands

```bash
# Build the image
make docker-build            # or: docker build -t genomics-ml-api .

# Start the API in a container (detached = background)
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down

# Run tests inside the container
docker run --rm genomics-ml-api pytest tests/ -v
```

## Volumes

- `./models:/app/models` — trained model files are read from host, so you don't rebuild the image after training
- `./ml_metadata.db:/app/ml_metadata.db` — SQLite DB persists across container restarts

## Notes

- The image is for **API serving only** — not for training.
  Training runs locally via `make run-pipeline`.
- If you add a new dependency, rebuild the image (volumes don't reinstall packages).
