"use client";

import { use, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import type { EChartsOption } from "echarts";
import { api, type JointRow, type TimelineRow } from "../../../../../lib/api";
import { Card, ErrorState, Loading } from "../../../../../components/ui";
import { Chart } from "../../../../../components/chart";

export default function BodyMotionPage({ params }: { params: Promise<{ person: string; session: string }> }) {
  const { person, session } = use(params);
  const [frame, setFrame] = useState(0);
  const [threshold, setThreshold] = useState(0.5);
  const [points, setPoints] = useState(true);
  const [skeleton, setSkeleton] = useState(true);

  const info = useQuery({ queryKey: ["v2info", person, session], queryFn: () => api.v2Info(person, session) });
  const timeline = useQuery({
    queryKey: ["v2timeline", person, session],
    queryFn: () => api.v2Timeline(person, session),
    enabled: info.isSuccess,
  });
  const quality = useQuery({
    queryKey: ["v2quality", person, session, frame, threshold],
    queryFn: () => api.v2Quality(person, session, frame, threshold),
    enabled: info.isSuccess,
  });

  const frames = info.data?.frames ?? 0;
  const row: TimelineRow | undefined = timeline.data?.[frame];

  // Playback steps through frames at the recording's frame rate, and stops at the last frame.
  const [playing, setPlaying] = useState(false);
  const fps = info.data?.fps ?? 0;
  const stepMs = fps > 0 ? Math.max(1000 / fps, 100) : 200;
  useEffect(() => {
    if (!playing || frames === 0) return;
    const id = window.setInterval(() => {
      setFrame((f) => {
        if (f >= frames - 1) {
          setPlaying(false);
          return f;
        }
        return f + 1;
      });
    }, stepMs);
    return () => window.clearInterval(id);
  }, [playing, frames, stepMs]);

  const timelineOption = useMemo<EChartsOption | null>(() => {
    if (!timeline.data) return null;
    const t = timeline.data.map((r) => Number(r.timestamp_ms) / 1000);
    return {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "time (s)" },
      yAxis: { type: "value", min: 0, max: 1.05, name: "detected (1 = yes)" },
      series: [
        { name: "face detected", type: "line", step: "end", showSymbol: false, data: t.map((x, i) => [x, timeline.data[i].face_detected ? 1 : 0]) },
        { name: "pose detected", type: "line", step: "end", showSymbol: false, data: t.map((x, i) => [x, timeline.data[i].pose_detected ? 1 : 0]) },
      ],
    };
  }, [timeline.data]);

  const pct = (v: number | null | undefined) => (v == null ? "n/a" : `${(v * 100).toFixed(1)} %`);

  return (
    <div className="flex flex-col gap-6">
      <header>
        <p className="text-sm text-[var(--muted)]">
          <Link href={`/people/${encodeURIComponent(person)}`} className="underline">{person}</Link> / {session}
        </p>
        <h1 className="text-2xl font-semibold">Body motion</h1>
        <p className="text-sm text-[var(--muted)]">
          Existing Pose Landmarker output. Tracking figures only; they do not measure the accuracy of the motion.
          Nothing is generated here.
        </p>
      </header>

      {info.isPending ? <Loading label="Checking pose data…" /> : null}
      {info.isError ? <ErrorState message={info.error.message} code={(info.error as { code?: string }).code} /> : null}

      {info.data && frames > 0 ? (
        <>
          <Card title="Frame and overlay">
            <div className="grid gap-4 md:grid-cols-[1fr_260px]">
              <div className="flex flex-col gap-3">
                <label className="flex flex-col gap-1 text-sm">
                  Frame: {frame} of {frames - 1}
                  <input type="range" min={0} max={frames - 1} value={frame} onChange={(e) => setFrame(Number(e.target.value))} aria-label="Frame" />
                </label>
                <div className="flex gap-2 text-sm">
                  <button type="button" className="rounded border border-[var(--border)] px-3 py-1" disabled={frame <= 0} onClick={() => { setPlaying(false); setFrame(frame - 1); }}>Previous</button>
                  <button type="button" className="rounded border border-[var(--border)] px-3 py-1" disabled={frame >= frames - 1} onClick={() => { setPlaying(false); setFrame(frame + 1); }}>Next</button>
                  <button type="button" className="rounded border border-[var(--border)] px-3 py-1" onClick={() => { if (!playing && frame >= frames - 1) setFrame(0); setPlaying(!playing); }} aria-pressed={playing}>{playing ? "Pause" : "Play"}</button>
                </div>
                {/* Served by the API from the stored recording; the image is not generated in the browser. */}
                <img
                  src={api.v2OverlayUrl(person, session, frame, threshold, points, skeleton)}
                  alt={`Pose overlay for frame ${frame}`}
                  className="w-full rounded border border-[var(--border)]"
                />
                <p className="text-xs text-[var(--muted)]">
                  Green: shown (visibility at or above the threshold, inside the frame). Grey: below the threshold.
                  Off-frame and missing joints are not drawn.
                </p>
              </div>
              <div className="flex flex-col gap-4 text-sm">
                <label className="flex flex-col gap-1">
                  Visibility threshold: {threshold.toFixed(2)}
                  <input type="range" min={0} max={1} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} aria-label="Visibility threshold" />
                </label>
                <label className="flex items-center gap-2"><input type="checkbox" checked={points} onChange={(e) => setPoints(e.target.checked)} />Show joints</label>
                <label className="flex items-center gap-2"><input type="checkbox" checked={skeleton} onChange={(e) => setSkeleton(e.target.checked)} />Show skeleton</label>
                <dl className="grid grid-cols-2 gap-3">
                  <div><dt className="text-[var(--muted)]">Timestamp</dt><dd className="font-semibold">{row ? `${(Number(row.timestamp_ms) / 1000).toFixed(3)} s` : "n/a"}</dd></div>
                  <div><dt className="text-[var(--muted)]">Face detected</dt><dd className="font-semibold">{row ? (row.face_detected ? "yes" : "no") : "n/a"}</dd></div>
                  <div><dt className="text-[var(--muted)]">Pose detected</dt><dd className="font-semibold">{row ? (row.pose_detected ? "yes" : "no") : "n/a"}</dd></div>
                  <div><dt className="text-[var(--muted)]">Video size</dt><dd className="font-semibold">{info.data.width}×{info.data.height}</dd></div>
                </dl>
              </div>
            </div>
          </Card>

          {timelineOption ? (
            <Card title="Detection timeline (timeline.csv)">
              <Chart option={timelineOption} height={240} label="Face and pose detection over time" />
            </Card>
          ) : null}

          <Card title="Pose tracking quality">
            {quality.isPending ? <Loading /> : null}
            {quality.isError ? <ErrorState message={quality.error.message} /> : null}
            {quality.data ? (
              <>
                <dl className="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
                  <div><dt className="text-[var(--muted)]">Pose detection rate</dt><dd className="text-xl font-semibold">{pct(quality.data.quality.pose_detection_rate)}</dd></div>
                  <div><dt className="text-[var(--muted)]">Frames without pose</dt><dd className="text-xl font-semibold">{quality.data.quality.missing_pose_frames.length}</dd></div>
                  <div><dt className="text-[var(--muted)]">Median visibility</dt><dd className="text-xl font-semibold">{quality.data.quality.median_visibility == null ? "n/a" : quality.data.quality.median_visibility.toFixed(2)}</dd></div>
                  <div><dt className="text-[var(--muted)]">Valid joints (frame {frame})</dt><dd className="text-xl font-semibold">{quality.data.quality.valid_joints_selected_frame} / 33</dd></div>
                </dl>
                <p className="mt-3 text-xs text-[var(--muted)]">
                  Joints shown across detected frames at threshold {threshold.toFixed(2)}: {pct(quality.data.quality.valid_joint_share_detected_frames)}. {quality.data.quality.scope}.
                </p>
              </>
            ) : null}
          </Card>

          <Card title={`Joints for frame ${frame}`}>
            {quality.data ? <JointTable rows={quality.data.joints} /> : <Loading />}
          </Card>
        </>
      ) : null}
    </div>
  );
}

function JointTable({ rows }: { rows: JointRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-[var(--muted)]">
          <tr>
            <th scope="col" className="py-2 font-medium">#</th>
            <th scope="col" className="py-2 font-medium">Joint</th>
            <th scope="col" className="py-2 font-medium">x</th>
            <th scope="col" className="py-2 font-medium">y</th>
            <th scope="col" className="py-2 font-medium">z</th>
            <th scope="col" className="py-2 font-medium">Visibility</th>
            <th scope="col" className="py-2 font-medium">State</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.index} className="border-t border-[var(--border)]">
              <td className="py-1">{r.index}</td>
              <td className="py-1">{r.joint}</td>
              <td className="py-1">{r.x == null ? "n/a" : r.x.toFixed(3)}</td>
              <td className="py-1">{r.y == null ? "n/a" : r.y.toFixed(3)}</td>
              <td className="py-1">{r.z == null ? "n/a" : r.z.toFixed(3)}</td>
              <td className="py-1">{r.visibility == null ? "n/a" : r.visibility.toFixed(2)}</td>
              <td className="py-1">{r.state.replace("_", " ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
