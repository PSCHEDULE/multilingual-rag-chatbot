"""Backward-compatible entrypoint; prefer `uv run uvicorn app.main:app`."""

from app.main import run

if __name__ == "__main__":
    run()
