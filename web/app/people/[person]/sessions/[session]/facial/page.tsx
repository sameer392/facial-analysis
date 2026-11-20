"use client";

import { use, useMemo, useState } from "react";
import Link from "next/link";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { EChartsOption } from "echarts";
import { api } from "../../../../../lib/api";
import { Card, ErrorState, Loading } from "../../../../../components/ui";
import { Chart } from "../../../../../components/chart";

const MOTION_FIELDS = ["velocity", "acceleration", "displacement", "neutral_offset"];
const EXPRESSION_FIELDS = ["jaw_open", "mouth_activity", "eye_blink"];
const HEAD_FIELDS = ["yaw_deg", "pitch_deg", "roll_deg"];
const TIMELINE_FIELDS = ["face_detected", "is_neutral"];
const ALL_FIELDS = [...TIMELINE_FIELDS, ...MOTION_FIELDS, ...EXPRESSION_FIELDS, ...HEAD_FIELDS];

const OVERLAY_MODES: [string, string][] = [
  ["hybrid", "Hybrid (faint mesh + contours)"],
  ["wireframe", "Wireframe (full mesh)"],
  ["contours", "Contours (eyes, brows, nose, lips, oval)"],
  ["points", "Points (original dots)"],
  ["off", "Off"],
];

const REGION_GROUPS: [string, string[]][] = [
  ["Mouth", ["jawOpen", "mouthPucker", "mouthFunnel", "mouthSmileLeft", "mouthSmileRight"]],
  ["Eyes", ["eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight"]],
  ["Eyebrows", ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight"]],
];
const REGION_NAMES = REGION_GROUPS.flatMap(([, names]) => names);

const DEFAULT_LANDMARK_IDS = [1, 13, 14, 33, 61, 199, 263, 291];

function fmt(v: number | null | undefined, digits = 5): string {
  return v == null ? "n/a" : v.toFixed(digits);
}

function LandmarkTable({ ids, raw, comp }: { ids: number[]; raw: (number | null)[][] | undefined; comp: (number | null)[][] | undefined }) {
  const row = (i: number) => (
    <tr key={i} className="border-t border-[var(--border)]">
      <td className="py-1 pr-3">{i}</td>
      <td className="py-1 pr-3">{fmt(raw?.[i]?.[0])}</td>
      <td className="py-1 pr-3">{fmt(raw?.[i]?.[1])}</td>
      <td className="py-1 pr-3">{fmt(raw?.[i]?.[2])}</td>
      <td className="py-1 pr-3">{fmt(comp?.[i]?.[0])}</td>
      <td className="py-1 pr-3">{fmt(comp?.[i]?.[1])}</td>
      <td className="py-1">{fmt(comp?.[i]?.[2])}</td>
    </tr>
  );
  const head = (
    <thead className="text-left text-[var(--muted)]">
      <tr>
        <th scope="col" className="py-2 pr-3 font-medium">ID</th>
        <th scope="col" className="py-2 pr-3 font-medium">Raw x</th>
        <th scope="col" className="py-2 pr-3 font-medium">Raw y</th>
        <th scope="col" className="py-2 pr-3 font-medium">Raw z</th>
        <th scope="col" className="py-2 pr-3 font-medium">Compensated x</th>
        <th scope="col" className="py-2 pr-3 font-medium">Compensated y</th>
        <th scope="col" className="py-2 font-medium">Compensated z</th>
      </tr>
    </thead>
  );
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        {head}
        <tbody>{ids.map(row)}</tbody>
      </table>
    </div>
  );
}

const SECTIONS: [string, string][] = [
  ["summary", "Summary"],
  ["frame", "Video frame"],
  ["timeline", "Detection timeline"],
  ["motion", "Motion"],
  ["expressions", "Expressions and blinks"],
  ["regions", "Movement by region"],
  ["head", "Head rotation"],
  ["landmarks", "Landmarks for one frame"],
];

function lineOption(time: (number | null)[], series: [string, (number | boolean | null)[]][]): EChartsOption {
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0 },
    grid: { left: 50, right: 20, top: 40, bottom: 40 },
    xAxis: { type: "value", name: "time (s)" },
    yAxis: { type: "value" },
    series: series.map(([name, values]) => ({
      name,
      type: "line",
      showSymbol: false,
      connectNulls: false,
      data: values.map((v, i): [number | null, number | null] => [time[i], typeof v === "boolean" ? (v ? 1 : 0) : v]),
    })),
  };
}

function landmarkOption(points: (number | null)[][] | undefined, title: string): EChartsOption {
  return {
    title: { text: title, left: "center", textStyle: { fontSize: 13 } },
    grid: { left: 40, right: 20, top: 40, bottom: 30 },
    xAxis: { type: "value", show: false },
    yAxis: { type: "value", show: false, inverse: true },
    series: [{
      type: "scatter",
      symbolSize: 3,
      data: (points ?? []).filter((p) => p[0] != null && p[1] != null).map((p) => [p[0], p[1]]),
    }],
  };
}

export default function FacePage({ params }: { params: Promise<{ person: string; session: string }> }) {
  const { person, session } = use(params);
  const [frame, setFrame] = useState(0);
  const [mode, setMode] = useState("hybrid");
  const [opacity, setOpacity] = useState(0.8);
  const [thickness, setThickness] = useState(1);
  const [pointSize, setPointSize] = useState(2);
  const [landmarkIds, setLandmarkIds] = useState<number[]>(DEFAULT_LANDMARK_IDS);

  const summary = useQuery({ queryKey: ["face-summary", person, session], queryFn: () => api.faceSummary(person, session) });
  const video = useQuery({ queryKey: ["face-video", person, session], queryFn: () => api.faceVideo(person, session) });
  const series = useQuery({ queryKey: ["face-series", person, session], queryFn: () => api.faceSeries(person, session, ALL_FIELDS) });
  const blinks = useQuery({ queryKey: ["face-blinks", person, session], queryFn: () => api.faceBlinks(person, session) });
  const raw = useQuery({
    queryKey: ["face-landmarks", person, session, frame, "raw"],
    queryFn: () => api.faceLandmarks(person, session, frame, false),
    enabled: summary.isSuccess,
    placeholderData: keepPreviousData,
  });
  const compensated = useQuery({
    queryKey: ["face-landmarks", person, session, frame, "comp"],
    queryFn: () => api.faceLandmarks(person, session, frame, true),
    enabled: summary.isSuccess,
    placeholderData: keepPreviousData,
  });
  const regionSeries = useQuery({
    queryKey: ["face-blendshape-series", person, session],
    queryFn: () => api.faceBlendshapeSeries(person, session, REGION_NAMES),
    enabled: summary.isSuccess,
  });
  const blendFrame = useQuery({
    queryKey: ["face-blendshapes-frame", person, session, frame],
    queryFn: () => api.faceBlendshapesFrame(person, session, frame),
    enabled: summary.isSuccess,
    placeholderData: keepPreviousData,
  });

  const topBlendshapes = useMemo<EChartsOption | null>(() => {
    if (!blendFrame.data) return null;
    const top = blendFrame.data.values.slice(0, 15).reverse();
    return {
      title: { text: "Top 15 blendshape scores for this frame", left: "center", textStyle: { fontSize: 13 } },
      grid: { left: 150, right: 30, top: 40, bottom: 30 },
      xAxis: { type: "value", min: 0, max: 1, name: "score" },
      yAxis: { type: "category", data: top.map(([name]) => name), axisLabel: { fontSize: 11 } },
      series: [{ type: "bar", data: top.map(([, score]) => score) }],
    };
  }, [blendFrame.data]);

  const timelineOption = useMemo<EChartsOption | null>(() => {
    if (!series.data) return null;
    return lineOption(series.data.time_s, [
      ["face detected", series.data.face_detected],
      ["neutral frame", series.data.is_neutral],
    ]);
  }, [series.data]);

  const motionOption = useMemo<EChartsOption | null>(() => {
    if (!series.data) return null;
    return lineOption(series.data.time_s, [
      ["velocity", series.data.velocity],
      ["acceleration", series.data.acceleration],
      ["displacement", series.data.displacement],
      ["neutral offset", series.data.neutral_offset],
    ]);
  }, [series.data]);

  const expressionOption = useMemo<EChartsOption | null>(() => {
    if (!series.data) return null;
    return lineOption(series.data.time_s, [
      ["jaw open", series.data.jaw_open],
      ["mouth activity", series.data.mouth_activity],
      ["eye blink", series.data.eye_blink],
    ]);
  }, [series.data]);

  const headOption = useMemo<EChartsOption | null>(() => {
    if (!series.data) return null;
    return lineOption(series.data.time_s, [
      ["yaw", series.data.yaw_deg],
      ["pitch", series.data.pitch_deg],
      ["roll", series.data.roll_deg],
    ]);
  }, [series.data]);

  const q = summary.data?.quality;
  const detectedFrames = series.data ? series.data.face_detected.filter((v) => v === true).length : null;
  const faceNow = series.data ? series.data.face_detected[frame] === true : null;
  const timeNow = series.data ? series.data.time_s[frame] : null;
  const landmarksNow = raw.data ? raw.data.points.filter((p) => p[0] != null).length : null;
  const pct = (v: number | null | undefined) => (v == null ? "n/a" : `${(v * 100).toFixed(1)} %`);
  const frames = summary.data?.frames ?? 1;

  return (
    <div className="flex flex-col gap-6">
      <header>
        <p className="text-sm text-[var(--muted)]">
          <Link href={`/people/${encodeURIComponent(person)}`} className="underline">{person}</Link> / {session}
        </p>
        <h1 className="text-2xl font-semibold">Face analysis</h1>
        <p className="text-sm text-[var(--muted)]">
          Tracking figures and motion values from the face analysis. Tracking figures do not measure motion accuracy.
        </p>
        <nav aria-label="On this page" className="mt-3 flex flex-wrap gap-3 text-sm">
          {SECTIONS.map(([id, label]) => (
            <a key={id} href={`#${id}`} className="underline">{label}</a>
          ))}
        </nav>
      </header>

      <section id="summary" className="scroll-mt-6">
        {summary.isPending ? <Loading label="Loading summary…" /> : null}
        {summary.isError ? <ErrorState message={summary.error.message} code={(summary.error as { code?: string }).code} /> : null}
        {video.data ? (
          <Card title="Video">
            <dl className="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
              <div><dt className="text-[var(--muted)]">Video FPS</dt><dd className="text-xl font-semibold">{video.data.fps ? video.data.fps.toFixed(2) : "Unknown"}</dd></div>
              <div><dt className="text-[var(--muted)]">Video frames</dt><dd className="text-xl font-semibold">{video.data.frames.toLocaleString()}</dd></div>
              <div><dt className="text-[var(--muted)]">Duration</dt><dd className="text-xl font-semibold">{video.data.duration_s == null ? "Unknown" : `${video.data.duration_s.toFixed(2)} sec`}</dd></div>
              <div><dt className="text-[var(--muted)]">Frames with face</dt><dd className="text-xl font-semibold">{detectedFrames == null ? "n/a" : `${detectedFrames.toLocaleString()} / ${(series.data?.time_s.length ?? 0).toLocaleString()}`}</dd></div>
            </dl>
          </Card>
        ) : null}
        {q ? (
          <Card title="Summary and tracking quality">
            <dl className="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
              <div><dt className="text-[var(--muted)]">Frames</dt><dd className="text-xl font-semibold">{summary.data?.frames}</dd></div>
              <div><dt className="text-[var(--muted)]">Face detection</dt><dd className="text-xl font-semibold">{pct(q.detection_rate)}</dd></div>
              <div><dt className="text-[var(--muted)]">Quality score</dt><dd className="text-xl font-semibold">{q.quality_score ?? "n/a"} / 100</dd></div>
              <div><dt className="text-[var(--muted)]">Blink events</dt><dd className="text-xl font-semibold">{summary.data?.blink_events}</dd></div>
              <div><dt className="text-[var(--muted)]">Neutral frames</dt><dd className="text-xl font-semibold">{summary.data?.neutral_frames}</dd></div>
              <div><dt className="text-[var(--muted)]">Longest gap</dt><dd className="text-xl font-semibold">{q.longest_missing_run_frames} frames</dd></div>
              <div><dt className="text-[var(--muted)]">Median jitter</dt><dd className="text-xl font-semibold">{q.median_landmark_jitter == null ? "n/a" : q.median_landmark_jitter.toFixed(5)}</dd></div>
              <div><dt className="text-[var(--muted)]">Interruptions</dt><dd className="text-xl font-semibold">{q.interruption_count}</dd></div>
            </dl>
            <p className="mt-3 text-xs text-[var(--muted)]">{q.scope}</p>
            {q.interruptions.length > 0 ? (
              <ul className="mt-3 list-disc pl-5 text-sm">
                {q.interruptions.map((r) => (
                  <li key={r.start_frame}>frames {r.start_frame} to {r.end_frame} ({r.length_frames} frames without a face)</li>
                ))}
              </ul>
            ) : null}
          </Card>
        ) : null}
      </section>

      <section id="frame" className="scroll-mt-6">
        <Card title={`Video frame ${frame} and landmark overlay`}>
          <div className="mb-3 grid gap-3 text-sm md:grid-cols-5">
            <label className="flex flex-col gap-1 md:col-span-2">
              Landmark overlay
              <select value={mode} onChange={(e) => setMode(e.target.value)} className="rounded border border-[var(--border)] bg-transparent px-2 py-1">
                {OVERLAY_MODES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              Overlay opacity: {opacity.toFixed(2)}
              <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
            </label>
            <label className="flex flex-col gap-1">
              Line thickness: {thickness}
              <input type="range" min={1} max={3} step={1} value={thickness} onChange={(e) => setThickness(Number(e.target.value))} />
            </label>
            <label className="flex flex-col gap-1">
              Point size: {pointSize}
              <input type="range" min={1} max={5} step={1} value={pointSize} onChange={(e) => setPointSize(Number(e.target.value))} />
            </label>
          </div>
          <label className="mb-3 flex max-w-md flex-col gap-1 text-sm">
            Frame: {frame} of {Math.max(frames - 1, 0)}
            <input type="range" min={0} max={Math.max(frames - 1, 0)} value={frame} onChange={(e) => setFrame(Number(e.target.value))} aria-label="Frame" />
          </label>
          <div className="grid gap-4 md:grid-cols-2">
            <figure>
              <figcaption className="mb-1 text-sm text-[var(--muted)]">Original frame</figcaption>
              <img src={api.faceOriginalUrl(person, session, frame)} alt={`Original frame ${frame}`} className="w-full rounded border border-[var(--border)]" />
            </figure>
            <figure>
              <figcaption className="mb-1 text-sm text-[var(--muted)]">Landmark overlay</figcaption>
              <img src={api.faceOverlayUrl(person, session, frame, mode, opacity, thickness, pointSize)} alt={`Landmark overlay for frame ${frame}`} className="w-full rounded border border-[var(--border)]" />
            </figure>
          </div>
          <p className="mt-2 text-sm">
            Frame {frame}{timeNow == null ? "" : ` · ${timeNow.toFixed(3)} seconds`} · Face detected: {faceNow == null ? "n/a" : faceNow ? "yes" : "no"}
            {landmarksNow == null ? "" : ` · Landmarks: ${landmarksNow}`}
          </p>
          <p className="mt-1 text-xs text-[var(--muted)]">The same frame is used for the landmark views further down.</p>
        </Card>
        {topBlendshapes ? (
          <div className="mt-6">
            <Card title="Blendshape values at selected frame">
              <Chart option={topBlendshapes} height={480} label="Top 15 blendshape scores for the selected frame" />
              {blendFrame.data && blendFrame.data.values.length === 0 ? (
                <p className="mt-2 text-sm text-[var(--muted)]">No blendshape scores for this frame.</p>
              ) : null}
            </Card>
          </div>
        ) : null}
      </section>

      <section id="timeline" className="scroll-mt-6">
        {timelineOption ? (
          <Card title="Face detection and neutral frames over time">
            <Chart option={timelineOption} height={200} label="Face detection and neutral frames over time" />
          </Card>
        ) : series.isPending ? <Loading label="Loading series…" /> : null}
        {series.isError ? <ErrorState message={series.error.message} /> : null}
      </section>

      <section id="motion" className="scroll-mt-6">
        {motionOption ? (
          <Card title="Motion over time (velocity, acceleration, displacement, neutral offset)">
            <Chart option={motionOption} label="Motion over time" />
          </Card>
        ) : null}
      </section>

      <section id="expressions" className="scroll-mt-6 flex flex-col gap-6">
        {expressionOption ? (
          <Card title="Mouth and eye blendshapes">
            <Chart option={expressionOption} label="Mouth and eye blendshapes over time" />
          </Card>
        ) : null}
        <Card title="Blink events">
          {blinks.isPending ? <Loading /> : null}
          {blinks.data && blinks.data.length === 0 ? <p className="text-sm text-[var(--muted)]">No blinks detected at the 0.5 threshold.</p> : null}
          {blinks.data && blinks.data.length > 0 ? (
            <table className="w-full text-left text-sm">
              <thead className="text-[var(--muted)]">
                <tr>
                  <th scope="col" className="py-2 font-medium">Start frame</th>
                  <th scope="col" className="py-2 font-medium">End frame</th>
                  <th scope="col" className="py-2 font-medium">Start (s)</th>
                  <th scope="col" className="py-2 font-medium">Duration (ms)</th>
                </tr>
              </thead>
              <tbody>
                {blinks.data.map((b) => (
                  <tr key={`${b.start_frame}-${b.end_frame}`} className="border-t border-[var(--border)]">
                    <td className="py-2">{b.start_frame}</td>
                    <td className="py-2">{b.end_frame}</td>
                    <td className="py-2">{b.start_s?.toFixed(3) ?? "n/a"}</td>
                    <td className="py-2">{b.duration_ms?.toFixed(0) ?? "n/a"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </Card>
      </section>

      <section id="regions" className="scroll-mt-6 flex flex-col gap-6">
        {regionSeries.isPending ? <Loading label="Loading blendshapes…" /> : null}
        {regionSeries.isError ? <ErrorState message={regionSeries.error.message} /> : null}
        <p className="text-sm">
          <a className="underline" href={api.faceBlendshapesCsvUrl(person, session)} download>Download all blendshape scores (CSV)</a>
        </p>
        {regionSeries.data ? REGION_GROUPS.map(([group, names]) => {
          const available = names.filter((n) => regionSeries.data.values[n]);
          if (available.length === 0) return null;
          const option: EChartsOption = {
            tooltip: { trigger: "axis" },
            legend: { top: 0, type: "scroll" },
            grid: { left: 50, right: 20, top: 40, bottom: 40 },
            xAxis: { type: "value", name: "time (s)" },
            yAxis: { type: "value", min: 0, max: 1 },
            series: available.map((n) => ({
              name: n,
              type: "line",
              showSymbol: false,
              data: regionSeries.data.values[n].map((v, i): [number | null, number | null] => [regionSeries.data.time_s[i], v]),
            })),
          };
          return (
            <Card key={group} title={`${group} movement over the complete recording`}>
              <Chart option={option} label={`${group} blendshapes over time`} />
            </Card>
          );
        }) : null}
      </section>

      <section id="head" className="scroll-mt-6">
        {headOption ? (
          <Card title="Head rotation (degrees, from the transformation matrix)">
            <Chart option={headOption} label="Head rotation over time" />
          </Card>
        ) : null}
      </section>

      <section id="landmarks" className="scroll-mt-6">
        <Card title="Landmarks for one frame">
          <div className="grid gap-4 md:grid-cols-2">
            {raw.isPending && !raw.data ? <Loading label="Loading raw landmarks…" /> : <Chart option={landmarkOption(raw.data?.points, "Raw (normalised image units)")} label="Raw landmarks" />}
            {compensated.isPending && !compensated.data ? <Loading label="Loading compensated landmarks…" /> : <Chart option={landmarkOption(compensated.data?.points, "Head-pose compensated (face-size units)")} label="Compensated landmarks" />}
          </div>
          <p className="mt-2 text-xs text-[var(--muted)]">Frames without a face have no points.</p>
        </Card>
        <Card title="Landmark coordinates for this frame">
          {raw.data && raw.data.points.length > 0 ? (
            <>
              <label className="mb-3 flex max-w-md flex-col gap-1 text-sm">
                Landmark IDs (hold Ctrl or Cmd to select several)
                <select
                  multiple
                  size={8}
                  value={landmarkIds.map(String)}
                  onChange={(e) => setLandmarkIds(Array.from(e.target.selectedOptions, (o) => Number(o.value)))}
                  className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
                >
                  {raw.data.points.map((_, i) => <option key={i} value={i}>{i}</option>)}
                </select>
              </label>
              <LandmarkTable ids={landmarkIds} raw={raw.data.points} comp={compensated.data?.points} />
              <details className="mt-4">
                <summary className="cursor-pointer text-sm underline">Show all {raw.data.points.length} landmarks</summary>
                <div className="mt-3 max-h-[480px] overflow-auto">
                  <LandmarkTable ids={raw.data.points.map((_, i) => i)} raw={raw.data.points} comp={compensated.data?.points} />
                </div>
              </details>
            </>
          ) : (
            <p className="text-sm text-[var(--muted)]">No landmarks in this frame.</p>
          )}
        </Card>
      </section>
    </div>
  );
}
