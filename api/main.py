"""FastAPI application for the Next.js migration (phase P0: read-only).

Run locally (Windows PowerShell, from the project folder):
    ./.venv/Scripts/uvicorn.exe api.main:app --host 127.0.0.1 --port 8000

Only GET routes exist in P0. The server binds to 127.0.0.1 by design. Nothing is written.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from api import catalog, v1_data, v2_data
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


# ---- phase P2: V1 data for the analysis pages (read-only) ----------------------------------

@app.get("/api/persons/{person}/sessions/{session}/v1/summary")
def v1_summary(person: str, session: str):
    return v1_data.summary(person, session)


@app.get("/api/persons/{person}/sessions/{session}/v1/series")
def v1_series(person: str, session: str, fields: str = ""):
    names = [f for f in fields.split(",") if f]
    if not names:
        raise ApiError("invalid_request", "name at least one field, for example fields=velocity,jaw_open", 400)
    return v1_data.series(person, session, names)


@app.get("/api/persons/{person}/sessions/{session}/v1/blinks")
def v1_blinks(person: str, session: str):
    return v1_data.blinks(person, session)


@app.get("/api/persons/{person}/sessions/{session}/v1/landmarks/{index}")
def v1_landmarks(person: str, session: str, index: int, compensated: bool = False):
    return v1_data.landmarks(person, session, index, compensated)


@app.get("/api/persons/{person}/comparison")
def comparison(person: str, sessions: str = ""):
    names = [s for s in sessions.split(",") if s]
    return v1_data.comparison(person, names)


# ---- phase P3: V2 Pose and head-pose data for the web pages (read-only) -------------------

@app.get("/api/persons/{person}/sessions/{session}/v2")
def v2_info(person: str, session: str):
    return v2_data.info(person, session)


@app.get("/api/persons/{person}/sessions/{session}/v2/timeline")
def v2_timeline(person: str, session: str):
    return v2_data.timeline(person, session)


@app.get("/api/persons/{person}/sessions/{session}/v2/pose")
def v2_pose(person: str, session: str, start: int = 0, end: int = 1, threshold: float = v2_data.DEFAULT_THRESHOLD):
    return v2_data.pose(person, session, start, end, threshold)


@app.get("/api/persons/{person}/sessions/{session}/v2/frames/{index}/quality")
def v2_quality(person: str, session: str, index: int, threshold: float = v2_data.DEFAULT_THRESHOLD):
    return v2_data.quality(person, session, index, threshold)


@app.get("/api/persons/{person}/sessions/{session}/v2/frames/{index}/overlay.jpg")
def v2_overlay(person: str, session: str, index: int, threshold: float = v2_data.DEFAULT_THRESHOLD,
               points: bool = True, skeleton: bool = True):
    data = v2_data.overlay_jpeg(person, session, index, threshold, points, skeleton)
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/persons/{person}/sessions/{session}/v1/head-pose")
def v1_head_pose(person: str, session: str):
    return v2_data.head_pose(person, session)
