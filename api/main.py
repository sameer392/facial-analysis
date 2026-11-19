"""FastAPI application for the Next.js migration.

Run locally (Windows PowerShell, from the project folder):
    ./.venv/Scripts/uvicorn.exe api.main:app --host 127.0.0.1 --port 8000

Read routes (phases P0 to P3) only read. The job routes (phase P4) are the only writes: they start the existing
V2 build command as a separate process, and require the API token in the X-Facial-Token header. The token is
printed when the server starts (or set FACIAL_API_TOKEN). The server binds to 127.0.0.1 by design.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from api import catalog, jobs, system, v1_data, v2_data
from api.catalog import ApiError


@asynccontextmanager
async def lifespan(_app):
    interrupted = jobs.recover_interrupted()
    if interrupted:
        print(f"Marked {len(interrupted)} unfinished job(s) as interrupted.")
    print(f"API token for mutating requests (X-Facial-Token): {jobs.service().token}")
    yield


app = FastAPI(
    title="Facial Analysis API",
    version=catalog.API_VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


def require_token(x_facial_token: str | None = Header(default=None)) -> None:
    if not jobs.token_ok(x_facial_token):
        raise ApiError("unauthorized", "a valid API token is required (X-Facial-Token)", 401)


class JobRequest(BaseModel):
    person: str
    session: str
    kind: str = "v2_build"
    regenerate: bool = False


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


# ---- phase P4: jobs (the only writes; token required) --------------------------------------

@app.post("/api/jobs", status_code=202, dependencies=[Depends(require_token)])
def create_job(body: JobRequest):
    return jobs.create_job(body.person, body.session, body.kind, body.regenerate)


@app.get("/api/jobs")
def list_jobs(limit: int = 50):
    return jobs.service().store.list(min(max(limit, 1), 200))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.service().store.get(job_id)
    if job is None:
        raise ApiError("not_found", "job not found", 404)
    return job


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(require_token)])
def cancel_job(job_id: str):
    return jobs.cancel_job(job_id)


# ---- phase P5: models, logs and settings (read-only) ----------------------------------------

@app.get("/api/models")
def models_status():
    return system.models()


@app.get("/api/logs/jobs")
def job_logs(limit: int = 50):
    return system.job_logs(limit)


@app.get("/api/logs/jobs/{job_id}")
def job_log(job_id: str, lines: int = 200):
    return system.job_log_tail(job_id, lines)


@app.get("/api/settings")
def settings():
    return system.settings()


# ---- phase P6: layers not built yet (read-only) ---------------------------------------------

@app.get("/api/layers")
def layers():
    return system.layers()
