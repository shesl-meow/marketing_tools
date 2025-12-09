from .core.api import app


def run() -> None:
    """Run the FastAPI app with uvicorn."""
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
