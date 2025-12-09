from typing import Dict

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Initialize the FastAPI application and attach shared routes."""
    app = FastAPI(title="Marketing Tools API")

    @app.get("/", summary="Root")
    def read_root() -> Dict[str, str]:
        """Simple welcome endpoint."""
        return {"message": "Welcome to Marketing Tools API"}

    @app.get("/health", summary="Health check")
    def health() -> Dict[str, str]:
        """Lightweight liveness endpoint."""
        return {"status": "ok"}

    return app


# Export a module-level app for ASGI servers.
app = create_app()
