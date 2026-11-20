"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type HandGestureEventRow, type HandGestureObservationRow } from "../lib/api";
import { fmt, loadRows } from "../lib/hand-data";
import { Card, Empty, ErrorState, Loading } from "./ui";

const CLASSES = ["open_palm", "closed_fist", "pointing", "thumbs_up", "peace", "unknown"] as const;
const PAGE = 50;

/**
 * Gesture results. Frame-level observations (one per linked hand per frame) and temporal gesture events (stable
 * runs of frames) are shown in separate tables. The geometric margin is a rule distance, not a probability.
 */
export function HandGesturePanel({ person, session }: { person: string; session: string }) {
  const observations = useQuery({
    queryKey: ["hand-gesture-obs-all", person, session],
    queryFn: () => loadRows((offset, limit) => api.handGestureObservations(person, session, offset, limit)),
  });
  const events = useQuery({
    queryKey: ["hand-gesture-events-all", person, session],
    queryFn: () => loadRows((offset, limit) => api.handGestureEvents(person, session, offset, limit)),
  });
  const quality = useQuery({
    queryKey: ["hand-quality-gestures", person, session],
    queryFn: () => api.handQualityGestures(person, session),
    retry: false,
  });

  const [gesture, setGesture] = useState<string>("");
  const [track, setTrack] = useState<string>("");
  const [fromS, setFromS] = useState<string>("");
  const [toS, setToS] = useState<string>("");
  const [page, setPage] = useState(0);

  const bounded = fromS !== "" || toS !== "";
  // True when a time is inside the chosen range. With no range set, every row passes.
  const inRange = (ms: number | null): boolean => {
    if (!bounded) return true;
    if (ms === null) return false;
    const s = ms / 1000;
    if (fromS !== "" && s < Number(fromS)) return false;
    if (toS !== "" && s > Number(toS)) return false;
    return true;
  };

  const obsRows = observations.data?.rows ?? [];
  const obsFiltered = useMemo(
    () => obsRows.filter((r) =>
      (gesture === "" || r.frame_class === gesture) &&
      (track === "" || String(r.track_id) === track) &&
      inRange(r.timestamp_ms)),
    [obsRows, gesture, track, fromS, toS], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const evRows = events.data?.rows ?? [];
  const evFiltered = useMemo(
    () => evRows.filter((e) =>
      (gesture === "" || e.gesture === gesture) &&
      (track === "" || String(e.track_id) === track) &&
      (!bounded || inRange(e.start_timestamp_ms) || inRange(e.end_timestamp_ms))),
    [evRows, gesture, track, fromS, toS], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const tracks = useMemo(() => [...new Set(obsRows.map((r) => r.track_id))].sort((a, b) => a - b), [obsRows]);
  const pages = Math.max(1, Math.ceil(obsFiltered.length / PAGE));
  const shown: HandGestureObservationRow[] = obsFiltered.slice(page * PAGE, page * PAGE + PAGE);

  return (
    <div className="flex flex-col gap-6">
      <Card title="Gesture filters">
        <div className="grid gap-3 text-sm md:grid-cols-4">
          <label className="flex flex-col gap-1">
            Gesture class
            <select value={gesture} onChange={(e) => { setGesture(e.target.value); setPage(0); }}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1">
              <option value="">All classes</option>
              {CLASSES.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            Track
            <select value={track} onChange={(e) => { setTrack(e.target.value); setPage(0); }}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1">
              <option value="">All tracks</option>
              {tracks.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            From (seconds)
            <input type="number" min={0} step={0.1} value={fromS} onChange={(e) => { setFromS(e.target.value); setPage(0); }}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1" />
          </label>
          <label className="flex flex-col gap-1">
            To (seconds)
            <input type="number" min={0} step={0.1} value={toS} onChange={(e) => { setToS(e.target.value); setPage(0); }}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1" />
          </label>
        </div>
        {observations.data?.truncated ? (
          <p className="mt-2 text-xs text-[var(--muted)]">Only the first {obsRows.length} of {observations.data.total} observations are loaded for filtering.</p>
        ) : null}
      </Card>

      <Card title="Classification summary (quality)">
        {quality.isPending ? <Loading label="Loading gesture quality…" /> : null}
        {quality.isError ? <p className="text-sm text-[var(--muted)]">The gesture summary is not in this session yet.</p> : null}
        {quality.data ? (
          <>
            <dl className="grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
              <div><dt className="text-[var(--muted)]">Observations</dt><dd>{quality.data.observations}</dd></div>
              <div><dt className="text-[var(--muted)]">Unknown</dt><dd>{quality.data.frame_classes.unknown ?? 0} ({fmt((100 * (quality.data.frame_classes.unknown ?? 0)) / Math.max(quality.data.observations, 1), 1)} %)</dd></div>
              <div><dt className="text-[var(--muted)]">Events</dt><dd>{quality.data.events}</dd></div>
              <div><dt className="text-[var(--muted)]">Median event (ms)</dt><dd>{fmt(quality.data.duration_ms.median, 0)}</dd></div>
            </dl>
            <p className="mt-2 text-sm">Unknown reasons: {Object.entries(quality.data.unknown_reasons).map(([k, v]) => `${k} ${v}`).join(", ") || "none"}</p>
            <p className="mt-1 text-xs text-[var(--muted)]">{quality.data.scope}</p>
          </>
        ) : null}
      </Card>

      <Card title="Gesture events (stable runs over time)">
        <p className="mb-2 text-xs text-[var(--muted)]">
          An event is a run of consecutive frames of one class on one track, with at least 5 supporting frames. The
          margin is a rule distance in reach units or degrees. It is not a probability, and it is not calibrated.
        </p>
        {events.isPending ? <Loading label="Loading events…" /> : null}
        {events.isError ? <ErrorState message={events.error.message} code={(events.error as { code?: string }).code} /> : null}
        {events.data && evFiltered.length === 0 ? <Empty title="No events match" detail="Try a different class, track or time range." /> : null}
        {evFiltered.length > 0 ? <EventTable rows={evFiltered} /> : null}
      </Card>

      <Card title="Frame-level observations">
        <p className="mb-2 text-xs text-[var(--muted)]">One row per linked hand per frame. Unknown rows say why.</p>
        {observations.isPending ? <Loading label="Loading observations…" /> : null}
        {observations.isError ? <ErrorState message={observations.error.message} code={(observations.error as { code?: string }).code} /> : null}
        {observations.data && obsFiltered.length === 0 ? <Empty title="No observations match" /> : null}
        {shown.length > 0 ? (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-[var(--muted)]">
                  <tr>
                    <th className="py-1 font-medium">Frame</th><th className="py-1 font-medium">Time (ms)</th>
                    <th className="py-1 font-medium">Track</th><th className="py-1 font-medium">Hand</th>
                    <th className="py-1 font-medium">Class</th><th className="py-1 font-medium">Reason</th>
                    <th className="py-1 font-medium">Quality</th><th className="py-1 font-medium">Index</th>
                    <th className="py-1 font-medium">Middle</th><th className="py-1 font-medium">Thumb</th>
                    <th className="py-1 font-medium">Geometric margin</th><th className="py-1 font-medium">Event</th>
                  </tr>
                </thead>
                <tbody>
                  {shown.map((r) => (
                    <tr key={`${r.track_id}-${r.frame}`} className="border-t border-[var(--border)]">
                      <td className="py-1">{r.frame}</td>
                      <td>{r.timestamp_ms ?? "n/a"}</td>
                      <td>{r.track_id}</td>
                      <td>{r.label}</td>
                      <td>{r.frame_class}</td>
                      <td>{r.reason ?? ""}</td>
                      <td>{r.quality}</td>
                      <td>{r.index_state ?? ""}</td>
                      <td>{r.middle_state ?? ""}</td>
                      <td>{r.thumb_state ?? ""}</td>
                      <td>{fmt(r.margin, 3)}</td>
                      <td>{r.event_id ?? ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-3 flex items-center gap-3 text-sm">
              <button type="button" disabled={page === 0} onClick={() => setPage((p) => p - 1)}
                className="rounded border border-[var(--border)] px-2 py-1 disabled:opacity-50">Previous</button>
              <span>Page {page + 1} of {pages} ({obsFiltered.length} rows)</span>
              <button type="button" disabled={page >= pages - 1} onClick={() => setPage((p) => p + 1)}
                className="rounded border border-[var(--border)] px-2 py-1 disabled:opacity-50">Next</button>
            </div>
          </>
        ) : null}
      </Card>
    </div>
  );
}

function EventTable({ rows }: { rows: HandGestureEventRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead className="text-[var(--muted)]">
          <tr>
            <th className="py-1 font-medium">Event</th><th className="py-1 font-medium">Gesture</th>
            <th className="py-1 font-medium">Track</th><th className="py-1 font-medium">Start (ms)</th>
            <th className="py-1 font-medium">End (ms)</th><th className="py-1 font-medium">Duration (ms)</th>
            <th className="py-1 font-medium">Supporting frames</th><th className="py-1 font-medium">Bridged</th>
            <th className="py-1 font-medium">Support quality</th><th className="py-1 font-medium">Geometric margin (median)</th>
            <th className="py-1 font-medium">Ended by</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((e) => (
            <tr key={e.event_id} className={`border-t border-[var(--border)] ${e.duration_ms < 1000 ? "text-[var(--muted)]" : ""}`}>
              <td className="py-1">{e.event_id}</td>
              <td>{e.gesture}{e.duration_ms < 1000 ? " (short)" : ""}</td>
              <td>{e.track_id}</td>
              <td>{e.start_timestamp_ms}</td>
              <td>{e.end_timestamp_ms}</td>
              <td>{e.duration_ms}</td>
              <td>{e.supporting_observations}</td>
              <td>{e.bridged_observations}</td>
              <td>{fmt(e.support_quality, 2)}</td>
              <td>{fmt(e.margin_median, 3)}</td>
              <td>{e.end_reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
