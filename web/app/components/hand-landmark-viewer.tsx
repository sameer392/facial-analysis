"use client";

import { useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api, type HandLandmarkPage, type HandPoint } from "../lib/api";
import { FINGERTIP_INDICES, HAND_CONNECTIONS, HAND_LANDMARK_NAMES, frameTimeMs, fmt } from "../lib/hand-data";
import { Card, Loading } from "./ui";

const SIDE_COLOUR = { left: "#f5a524", right: "#3b82f6" } as const;

/**
 * One frame of the recording: the original frame, with the landmarks the API returned for each hand drawn over it.
 * Points are normalised image coordinates, so they are drawn on a 0-to-1 box stretched over the frame. A hand with
 * no landmarks in this frame is not drawn; nothing is estimated.
 */
export function HandLandmarkViewer({
  person,
  session,
  totalFrames,
  fps,
}: {
  person: string;
  session: string;
  totalFrames: number;
  fps: number | null;
}) {
  const [frame, setFrame] = useState(0);
  const [showLeft, setShowLeft] = useState(true);
  const [showRight, setShowRight] = useState(true);
  const [serverOverlay, setServerOverlay] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);
  const last = Math.max(totalFrames - 1, 0);
  const left = useQuery({
    queryKey: ["hand-frame", person, session, "left", frame],
    queryFn: () => api.handLandmarks(person, session, "left", frame, 1),
    placeholderData: keepPreviousData,
  });
  const right = useQuery({
    queryKey: ["hand-frame", person, session, "right", frame],
    queryFn: () => api.handLandmarks(person, session, "right", frame, 1),
    placeholderData: keepPreviousData,
  });

  const clamp = (value: number) => Math.min(Math.max(value, 0), last);
  const time = frameTimeMs(frame, fps);

  return (
    <Card title={`Frame review · frame ${frame} of ${last}${time === null ? "" : ` · ${(time / 1000).toFixed(3)} s`}`}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <button type="button" onClick={() => setFrame((f) => clamp(f - 1))} disabled={frame <= 0}
          className="rounded border border-[var(--border)] px-2 py-1 disabled:opacity-50">Previous</button>
        <button type="button" onClick={() => setFrame((f) => clamp(f + 1))} disabled={frame >= last}
          className="rounded border border-[var(--border)] px-2 py-1 disabled:opacity-50">Next</button>
      </div>
      <input
        type="range"
        min={0}
        max={last}
        value={frame}
        onChange={(e) => setFrame(clamp(Number(e.target.value)))}
        aria-label="Frame"
        className="mb-4 w-full"
      />

      <div className="mb-3 flex flex-wrap gap-4 text-sm">
        <label className="flex items-center gap-2"><input type="checkbox" checked={showLeft} onChange={(e) => setShowLeft(e.target.checked)} /> Left hand</label>
        <label className="flex items-center gap-2"><input type="checkbox" checked={showRight} onChange={(e) => setShowRight(e.target.checked)} /> Right hand</label>
        <label className="flex items-center gap-2"><input type="checkbox" checked={serverOverlay} onChange={(e) => setServerOverlay(e.target.checked)} /> Show the server overlay instead</label>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <figure className="relative">
          {serverOverlay ? (
            <img
              src={api.handOverlayUrl(person, session, frame)}
              alt={`Frame ${frame} with hand landmarks drawn by the server`}
              className="w-full rounded border border-[var(--border)]"
              onError={() => setImageFailed(true)}
            />
          ) : (
            <img
              src={api.faceOriginalUrl(person, session, frame)}
              alt={`Original frame ${frame}`}
              className="w-full rounded border border-[var(--border)]"
              onError={() => setImageFailed(true)}
            />
          )}
          {!serverOverlay ? (
            <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
              {showLeft ? <HandShape page={left.data} colour={SIDE_COLOUR.left} /> : null}
              {showRight ? <HandShape page={right.data} colour={SIDE_COLOUR.right} /> : null}
            </svg>
          ) : null}
          {imageFailed ? <figcaption className="mt-1 text-sm text-[var(--muted)]">The frame image is not available for this frame.</figcaption> : null}
        </figure>

        <div className="flex flex-col gap-3 text-sm">
          <HandStatus label="Left hand" page={left} />
          <HandStatus label="Right hand" page={right} />
          <p className="text-xs text-[var(--muted)]">
            Detection status from the landmark extraction: valid, low_confidence (below 0.7 handedness score), or outside_frame
            (some landmarks are off the image). The model&apos;s score itself is not part of this API.
          </p>
          <FingerTable page={left.data} label="Left" />
          <FingerTable page={right.data} label="Right" />
        </div>
      </div>
    </Card>
  );
}

function HandShape({ page, colour }: { page: HandLandmarkPage | undefined; colour: string }) {
  const points = page?.frames[0]?.points;
  if (!points) return null;
  const at = (i: number): [number, number] | null => {
    const p: HandPoint = points[i];
    return p[0] !== null && p[1] !== null ? [p[0], p[1]] : null;
  };
  return (
    <g stroke={colour} fill={colour}>
      {HAND_CONNECTIONS.map(([a, b]) => {
        const pa = at(a);
        const pb = at(b);
        if (!pa || !pb) return null;
        return <line key={`${a}-${b}`} x1={pa[0]} y1={pa[1]} x2={pb[0]} y2={pb[1]} strokeWidth={0.004} />;
      })}
      {points.map((_, i) => {
        const p = at(i);
        if (!p) return null;
        const tip = (FINGERTIP_INDICES as readonly number[]).includes(i);
        return <circle key={i} cx={p[0]} cy={p[1]} r={tip ? 0.009 : 0.005} />;
      })}
    </g>
  );
}

function HandStatus({ label, page }: { label: string; page: { isPending: boolean; isError: boolean; error: unknown; data?: HandLandmarkPage } }) {
  if (page.isError) {
    return <p><span className="font-medium">{label}:</span> not available for this frame.</p>;
  }
  if (page.isPending && !page.data) return <Loading label={`Loading ${label.toLowerCase()}…`} />;
  const frame = page.data?.frames[0];
  if (!frame) return <p><span className="font-medium">{label}:</span> no data for this frame.</p>;
  return (
    <p>
      <span className="font-medium">{label}:</span>{" "}
      {frame.status === "not_detected" || frame.status === null ? "not detected in this frame" : `status ${frame.status}`}
    </p>
  );
}

function FingerTable({ page, label }: { page: HandLandmarkPage | undefined; label: string }) {
  const points = page?.frames[0]?.points;
  if (!points) return null;
  const picks = [0, ...FINGERTIP_INDICES];
  return (
    <div className="overflow-x-auto">
      <p className="mb-1 font-medium">{label} hand: wrist and fingertips (normalised image coordinates)</p>
      <table className="w-full text-left text-xs">
        <thead className="text-[var(--muted)]">
          <tr><th className="py-1 font-medium">Landmark</th><th className="py-1 font-medium">x</th><th className="py-1 font-medium">y</th><th className="py-1 font-medium">z</th></tr>
        </thead>
        <tbody>
          {picks.map((i) => (
            <tr key={i} className="border-t border-[var(--border)]">
              <td className="py-1">{HAND_LANDMARK_NAMES[i]} ({i})</td>
              <td>{fmt(points[i][0], 4)}</td>
              <td>{fmt(points[i][1], 4)}</td>
              <td>{fmt(points[i][2], 4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <details className="mt-2">
        <summary className="cursor-pointer text-xs underline">All 21 landmarks</summary>
        <table className="mt-1 w-full text-left text-xs">
          <tbody>
            {points.map((p, i) => (
              <tr key={i} className="border-t border-[var(--border)]">
                <td className="py-0.5">{i}: {HAND_LANDMARK_NAMES[i]}</td>
                <td>{fmt(p[0], 4)}</td>
                <td>{fmt(p[1], 4)}</td>
                <td>{fmt(p[2], 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

