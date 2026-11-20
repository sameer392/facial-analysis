"""FastAPI application for the Next.js migration.

Run locally (Windows PowerShell, from the project folder):
    ./.venv/Scripts/uvicorn.exe api.main:app --host 127.0.0.1 --port 8000

Read routes (phases P0 to P3) only read. The job routes (phase P4) are the only writes: they start the existing
pose data build command as a separate process, and require the API token in the X-Facial-Token header. The token is
kept in config/api_token.txt (created on first start; or set FACIAL_API_TOKEN). The server binds to 127.0.0.1 by design.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from api import catalog, face_data, job_events, jobs, manage, pose_data, system
from api.catalog import ApiError


@asynccontextmanager
async def lifespan(_app):
    interrupted = jobs.recover_interrupted()
    if interrupted:
        print(f"Marked {len(interrupted)} unfinished job(s) as interrupted.")
    print(f"API token file: {jobs.TOKEN_FILE} (enter this token once, on the Settings page)")
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
    kind: str = "pose_build"
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


# ---- phase P2: face data data for the analysis pages (read-only) ----------------------------------

@app.get("/api/persons/{person}/sessions/{session}/face/summary")
def face_summary(person: str, session: str):
    return face_data.summary(person, session)


@app.get("/api/persons/{person}/sessions/{session}/face/series")
def face_series(person: str, session: str, fields: str = ""):
    names = [f for f in fields.split(",") if f]
    if not names:
        raise ApiError("invalid_request", "name at least one field, for example fields=velocity,jaw_open", 400)
    return face_data.series(person, session, names)


@app.get("/api/persons/{person}/sessions/{session}/face/blinks")
def face_blinks(person: str, session: str):
    return face_data.blinks(person, session)


@app.get("/api/persons/{person}/sessions/{session}/face/landmarks/{index}")
def face_landmarks(person: str, session: str, index: int, compensated: bool = False):
    return face_data.landmarks(person, session, index, compensated)


@app.get("/api/persons/{person}/comparison")
def comparison(person: str, sessions: str = ""):
    names = [s for s in sessions.split(",") if s]
    return face_data.comparison(person, names)


# ---- phase P3: pose data and head-pose data for the web pages (read-only) -------------------

@app.get("/api/persons/{person}/sessions/{session}/pose")
def pose_info(person: str, session: str):
    return pose_data.info(person, session)


@app.get("/api/persons/{person}/sessions/{session}/pose/timeline")
def pose_timeline(person: str, session: str):
    return pose_data.timeline(person, session)


@app.get("/api/persons/{person}/sessions/{session}/pose/frames")
def pose_frames(person: str, session: str, start: int = 0, end: int = 1, threshold: float = pose_data.DEFAULT_THRESHOLD):
    return pose_data.pose(person, session, start, end, threshold)


@app.get("/api/persons/{person}/sessions/{session}/pose/frames/{index}/quality")
def pose_quality(person: str, session: str, index: int, threshold: float = pose_data.DEFAULT_THRESHOLD):
    return pose_data.quality(person, session, index, threshold)


@app.get("/api/persons/{person}/sessions/{session}/pose/frames/{index}/overlay.jpg")
def pose_overlay(person: str, session: str, index: int, threshold: float = pose_data.DEFAULT_THRESHOLD,
               points: bool = True, skeleton: bool = True):
    data = pose_data.overlay_jpeg(person, session, index, threshold, points, skeleton)
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/persons/{person}/sessions/{session}/face/head-pose")
def face_head_pose(person: str, session: str):
    return pose_data.head_pose(person, session)


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



# ---- management: people, recordings, datasets and job history (token and typed confirmation) ----

class PersonRequest(BaseModel):
    name: str


class ConfirmRequest(BaseModel):
    confirm: str | None = None


@app.post("/api/persons", status_code=201, dependencies=[Depends(require_token)])
def create_person(body: PersonRequest):
    return manage.create_person(body.name)


@app.post("/api/persons/{person}/delete", dependencies=[Depends(require_token)])
def delete_person(person: str, body: ConfirmRequest):
    return manage.delete_person(person, body.confirm)


@app.post("/api/persons/{person}/sessions/{session}/dataset/delete", dependencies=[Depends(require_token)])
def delete_dataset(person: str, session: str, body: ConfirmRequest):
    return manage.delete_dataset(person, session, body.confirm)


@app.post("/api/persons/{person}/sessions/{session}/video/delete", dependencies=[Depends(require_token)])
def delete_video(person: str, session: str, body: ConfirmRequest):
    return manage.delete_video(person, session, body.confirm)


@app.put("/api/persons/{person}/sessions/{session}/video", status_code=201, dependencies=[Depends(require_token)])
async def upload_video(person: str, session: str, request: Request):
    """Upload one MP4 as the raw body (Content-Type video/mp4). Refused if the session already has a video."""
    writer = manage.UploadWriter(person, session)
    try:
        async for chunk in request.stream():
            writer.write(chunk)
        return writer.finish()
    except BaseException:
        writer.abort()
        raise


@app.post("/api/jobs/history/clear", dependencies=[Depends(require_token)])
def clear_job_history(body: ConfirmRequest):
    return manage.clear_job_history(body.confirm)


class NewNameRequest(BaseModel):
    to: str


@app.post("/api/persons/{person}/rename", dependencies=[Depends(require_token)])
def rename_person(person: str, body: NewNameRequest):
    return manage.rename_person(person, body.to)


@app.post("/api/persons/{person}/sessions/{session}/rename", dependencies=[Depends(require_token)])
def rename_session(person: str, session: str, body: NewNameRequest):
    return manage.rename_session(person, session, body.to)


@app.get("/api/jobs/{job_id}/events")
async def job_progress_events(job_id: str):
    if jobs.service().store.get(job_id) is None:
        raise ApiError("not_found", "job not found", 404)
    return StreamingResponse(job_events.event_stream(job_id), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/face/topology")
def face_topology():
    return face_data.topology()


@app.get("/api/persons/{person}/sessions/{session}/face/frames/{index}/overlay.jpg")
def face_overlay(person: str, session: str, index: int, mode: str = "hybrid", opacity: float = 0.8,
                 thickness: int = 1, point_radius: int = 2):
    data = face_data.frame_image(person, session, index, mode, opacity, thickness, point_radius)
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/persons/{person}/sessions/{session}/face/frames/{index}/original.jpg")
def face_original(person: str, session: str, index: int):
    data = face_data.frame_image(person, session, index, "off", 0.0, 1, 1)
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/persons/{person}/sessions/{session}/face/frames/{index}/blendshapes")
def face_blendshapes_frame(person: str, session: str, index: int):
    return face_data.blendshape_frame(person, session, index)


@app.get("/api/persons/{person}/sessions/{session}/face/video")
def face_video(person: str, session: str):
    return face_data.video_info(person, session)
