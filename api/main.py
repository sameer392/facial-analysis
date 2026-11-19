"""FastAPI application for the Next.js migration (phase P0: read-only).

Run locally (Windows PowerShell, from the project folder):
    ./.venv/Scripts/uvicorn.exe api.main:app --host 127.0.0.1 --port 8000

Only GET routes exist in P0. The server binds to 127.0.0.1 by design. Nothing is written.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api import catalog
from api.catalog import ApiError

app = FastAPI(
    title="Facial Analysis API",
    version=catalog.API_VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url="/api/openapi.json",
)


@app.exception_handler(ApiError)
async def _api_error(_request, exc: ApiError):
    return JSONResponse(status_code=exc.status, content={"code": exc.code, "message": exc.message})


@app.get("/api/health")
def health():
    return catalog.health()


@app.get("/api/persons")
def persons():
    return catalog.list_persons()


@app.get("/api/persons/{person}/sessions")
def sessions(person: str):
    return catalog.list_sessions(person)
