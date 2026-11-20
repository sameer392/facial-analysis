"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type HandSummary, type HandValidation } from "../lib/api";
import { fmt } from "../lib/hand-data";
import { Badge, Card, ErrorState, Loading } from "./ui";

/** Quality of the hand dataset. Validation reads the whole session, so it runs only when asked, and is cached. */
export function HandQualityPanel({ person, session, summary }: { person: string; session: string; summary: HandSummary }) {
  const [validateRequested, setValidateRequested] = useState(false);
  const tracking = useQuery({
    queryKey: ["hand-quality-tracking", person, session],
    queryFn: () => api.handQualityTracking(person, session),
    retry: false,
  });
  const gestures = useQuery({
    queryKey: ["hand-quality-gestures", person, session],
    queryFn: () => api.handQualityGestures(person, session),
    retry: false,
  });
  const validation = useQuery({
    queryKey: ["hand-validation", person, session],
    queryFn: () => api.handValidation(person, session),
    enabled: validateRequested,
    staleTime: Infinity,
    retry: false,
  });

  const counts = summary.status_counts;
  const left = counts.left ?? {};
  const right = counts.right ?? {};
  const sum = (c: Record<string, number>, keys: string[]) => keys.reduce((a, k) => a + (c[k] ?? 0), 0);

  return (
    <div className="flex flex-col gap-6">
      <Card title="Detection">
        <dl className="grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
          <div><dt className="text-[var(--muted)]">Frames with a detected hand</dt><dd>{summary.detected_frames ?? "n/a"}</dd></div>
          <div><dt className="text-[var(--muted)]">Left: valid / low confidence</dt><dd>{left.valid ?? 0} / {left.low_confidence ?? 0}</dd></div>
          <div><dt className="text-[var(--muted)]">Right: valid / low confidence</dt><dd>{right.valid ?? 0} / {right.low_confidence ?? 0}</dd></div>
          <div><dt className="text-[var(--muted)]">Outside-frame (left / right)</dt><dd>{sum(left, ["outside_frame"])} / {sum(right, ["outside_frame"])}</dd></div>
          <div><dt className="text-[var(--muted)]">Left: not detected</dt><dd>{left.not_detected ?? 0}</dd></div>
          <div><dt className="text-[var(--muted)]">Right: not detected</dt><dd>{right.not_detected ?? 0}</dd></div>
          <div><dt className="text-[var(--muted)]">Unassigned (left / right)</dt><dd>{left.unassigned ?? 0} / {right.unassigned ?? 0}</dd></div>
        </dl>
        <p className="mt-2 text-xs text-[var(--muted)]">Low confidence means the handedness score was below 0.7. Outside-frame means some landmarks lie off the image; this is not an occlusion measure.</p>
      </Card>

      <Card title="Tracking">
        {tracking.isPending ? <Loading label="Loading tracking quality…" /> : null}
        {tracking.isError ? <p className="text-sm text-[var(--muted)]">Tracking has not run for this session.</p> : null}
        {tracking.data ? (
          <dl className="grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
            <div><dt className="text-[var(--muted)]">Tracks</dt><dd>{tracking.data.tracks_created}</dd></div>
            <div><dt className="text-[var(--muted)]">Interruptions</dt><dd>{tracking.data.interruptions}</dd></div>
            <div><dt className="text-[var(--muted)]">Reacquisitions</dt><dd>{tracking.data.reacquisitions}</dd></div>
            <div><dt className="text-[var(--muted)]">Uncertain links</dt><dd>{Object.values(tracking.data.uncertain_observations).reduce((a, b) => a + b, 0)}</dd></div>
            <div><dt className="text-[var(--muted)]">Observations</dt><dd>{tracking.data.observations}</dd></div>
            <div><dt className="text-[var(--muted)]">Irregular intervals</dt><dd>{tracking.data.timestamps.irregular_intervals}</dd></div>
            <div><dt className="text-[var(--muted)]">Frame index gaps</dt><dd>{tracking.data.timestamps.frame_index_gaps}</dd></div>
            <div><dt className="text-[var(--muted)]">Smoothing</dt><dd>{tracking.data.smoothing}</dd></div>
          </dl>
        ) : null}
        {tracking.data ? (
          <p className="mt-2 text-sm">Missing observation reasons: {Object.entries(tracking.data.missing_frame_reasons).map(([k, v]) => `${k} ${v}`).join(", ") || "none"}</p>
        ) : null}
      </Card>

      <Card title="Motion">
        {tracking.data ? (
          <>
            <dl className="grid grid-cols-2 gap-3 text-sm md:grid-cols-3">
              <div><dt className="text-[var(--muted)]">Velocity values</dt><dd>{tracking.data.velocity_px_per_s.count}</dd></div>
              <div><dt className="text-[var(--muted)]">Median velocity</dt><dd>{fmt(tracking.data.velocity_px_per_s.median, 1)} px/s</dd></div>
              <div><dt className="text-[var(--muted)]">Peak velocity</dt><dd>{fmt(tracking.data.velocity_px_per_s.peak, 1)} px/s</dd></div>
            </dl>
            <p className="mt-2 text-sm">Motion reasons: {Object.entries(tracking.data.motion_reasons).map(([k, v]) => `${k} ${v}`).join(", ")}</p>
            <p className="mt-1 text-xs text-[var(--muted)]">Coordinates: {tracking.data.coordinates}</p>
          </>
        ) : <p className="text-sm text-[var(--muted)]">Motion has not run for this session.</p>}
      </Card>

      <Card title="Gestures">
        {gestures.isPending ? <Loading label="Loading gesture quality…" /> : null}
        {gestures.isError ? <p className="text-sm text-[var(--muted)]">Gesture classification has not run for this session.</p> : null}
        {gestures.data ? (
          <dl className="grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
            <div><dt className="text-[var(--muted)]">Classified observations</dt><dd>{gestures.data.observations}</dd></div>
            <div><dt className="text-[var(--muted)]">Unknown</dt><dd>{gestures.data.frame_classes.unknown ?? 0}</dd></div>
            <div><dt className="text-[var(--muted)]">Events</dt><dd>{gestures.data.events}</dd></div>
            <div><dt className="text-[var(--muted)]">Event duration median (ms)</dt><dd>{fmt(gestures.data.duration_ms.median, 0)}</dd></div>
          </dl>
        ) : null}
      </Card>

      <Card title="Dataset validation">
        <p className="mb-3 text-sm text-[var(--muted)]">
          Checks that the landmark, tracking and gesture outputs agree with each other and with the manifest and metadata. It reads the
          whole session and changes nothing. Passing these checks shows the files are consistent; it does not show the
          results are accurate.
        </p>
        <button
          type="button"
          onClick={() => setValidateRequested(true)}
          disabled={validation.isFetching}
          className="rounded border border-[var(--border)] px-3 py-1 text-sm disabled:opacity-50"
        >
          {validation.data ? "Run validation again" : "Run validation"}
        </button>
        {validation.isFetching ? <div className="mt-3"><Loading label="Validating the session. This reads every file, so it can take a few seconds…" /></div> : null}
        {validation.isError ? <div className="mt-3"><ErrorState message={validation.error.message} code={(validation.error as { code?: string }).code} /></div> : null}
        {validation.data ? <ValidationResult report={validation.data} /> : null}
      </Card>

      <Card title="Extractor and model">
        <dl className="grid grid-cols-1 gap-3 text-sm md:grid-cols-2">
          <div><dt className="text-[var(--muted)]">Hand model file</dt><dd>{summary.models.hand.file ?? "not recorded"}</dd></div>
          <div><dt className="text-[var(--muted)]">Model version</dt><dd>{summary.models.hand.version ?? "not recorded"}</dd></div>
          <div><dt className="text-[var(--muted)]">Licence status (as recorded)</dt><dd>{summary.models.hand.licence_status ?? "not recorded"}. Not cleared for commercial use.</dd></div>
          <div><dt className="text-[var(--muted)]">Source video</dt><dd>{summary.source.width ?? "?"} × {summary.source.height ?? "?"} px, {summary.source.fps ?? "?"} fps, {summary.source.frame_count ?? "?"} frames</dd></div>
        </dl>
        {validation.data ? (
          <dl className="mt-3 grid grid-cols-1 gap-3 text-sm md:grid-cols-2">
            <div><dt className="text-[var(--muted)]">Processing time (UTC)</dt><dd>{validation.data.provenance.processing_timestamp_utc ?? "not recorded"}</dd></div>
            <div><dt className="text-[var(--muted)]">Model SHA-256</dt><dd className="break-all font-mono text-xs">{validation.data.provenance.hand_model?.sha256 ?? "not recorded"}</dd></div>
            <div><dt className="text-[var(--muted)]">Calibration</dt><dd>{validation.data.provenance.calibration ?? "none recorded"}</dd></div>
          </dl>
        ) : null}
      </Card>
    </div>
  );
}

function ValidationResult({ report }: { report: HandValidation }) {
  return (
    <div className="mt-3 flex flex-col gap-2 text-sm">
      <p>
        <Badge tone={report.valid ? "good" : "warn"}>{report.valid ? "No errors" : "Errors found"}</Badge>{" "}
        {report.errors.length} error(s), {report.warnings.length} warning(s)
      </p>
      {report.errors.length > 0 ? (
        <ul className="list-disc pl-5">{report.errors.map((e, i) => <li key={`e${i}`}>{e}</li>)}</ul>
      ) : null}
      {report.warnings.length > 0 ? (
        <ul className="list-disc pl-5 text-[var(--muted)]">{report.warnings.map((w, i) => <li key={`w${i}`}>{w}</li>)}</ul>
      ) : null}
    </div>
  );
}
