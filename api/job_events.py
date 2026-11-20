"""Live progress for one job, as a Server-Sent Events stream (read only, no token).

Each message is the job record as JSON. A message is sent when the state, stage or progress changes, and the
stream closes when the job is no longer active. Streams also close after max_seconds, and the browser reconnects.
"""

from __future__ import annotations

import asyncio
import json

from api import jobs

TERMINAL_OR_IDLE_POLL = 1.0
MAX_STREAM_SECONDS = 1800.0


async def event_stream(job_id: str, poll_seconds: float = TERMINAL_OR_IDLE_POLL,
                       max_seconds: float = MAX_STREAM_SECONDS):
    store = jobs.service().store
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_seconds
    last = None
    yield "retry: 2000\n\n"
    while True:
        job = store.get(job_id)
        if job is None:
            yield "event: missing\ndata: {}\n\n"
            return
        snapshot = (job["state"], job["stage"], job["progress"])
        if snapshot != last:
            last = snapshot
            yield f"data: {json.dumps(job)}\n\n"
        if job["state"] not in jobs.ACTIVE or loop.time() >= deadline:
            return
        await asyncio.sleep(poll_seconds)
