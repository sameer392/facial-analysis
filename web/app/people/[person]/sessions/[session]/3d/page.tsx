"use client";

import { use, useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import Link from "next/link";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { EChartsOption } from "echarts";
import { api } from "../../../../../lib/api";
import { Card, ErrorState, Loading } from "../../../../../components/ui";
import { Chart } from "../../../../../components/chart";

type View = "compensated" | "raw";
type Point = [number | null, number | null, number | null];

const PLAY_SPEEDS = [5, 10, 30] as const;

function linesOption(time: (number | null)[], series: [string, (number | null)[]][], yName: string): EChartsOption {
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0 },
    grid: { left: 50, right: 20, top: 40, bottom: 40 },
    xAxis: { type: "value", name: "time (s)" },
    yAxis: { type: "value", name: yName },
    series: series.map(([name, values]) => ({
      name,
      type: "line",
      showSymbol: false,
      data: values.map((v, i): [number | null, number | null] => [time[i], v]),
    })),
  };
}

/**
 * Turns the landmarks in 3D (yaw about the vertical axis, pitch about the horizontal axis, roll about the line
 * of sight), then projects them onto the plane. The maths is done here, so no extra library is needed.
 */
function rotatedPoints(points: Point[], yawDeg: number, pitchDeg: number, rollDeg: number) {
  const valid = points.filter((p): p is [number, number, number] => p[0] != null && p[1] != null && p[2] != null);
  if (valid.length === 0) return { xy: [] as [number, number][], depth: [] as number[] };
  const cx = valid.reduce((a, p) => a + p[0], 0) / valid.length;
  const cy = valid.reduce((a, p) => a + p[1], 0) / valid.length;
  const cz = valid.reduce((a, p) => a + p[2], 0) / valid.length;
  const yaw = (yawDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;
  const roll = (rollDeg * Math.PI) / 180;
  const cyw = Math.cos(yaw);
  const syw = Math.sin(yaw);
  const cpt = Math.cos(pitch);
  const spt = Math.sin(pitch);
  const cr = Math.cos(roll);
  const sr = Math.sin(roll);
  const xy: [number, number][] = [];
  const depth: number[] = [];
  for (const [x0, y0, z0] of valid) {
    const x = x0 - cx;
    const y = y0 - cy;
    const z = z0 - cz;
    const x1 = x * cyw + z * syw; // yaw
    const z1 = -x * syw + z * cyw;
    const y2 = y * cpt - z1 * spt; // pitch
    const z2 = y * spt + z1 * cpt;
    const x3 = x1 * cr - y2 * sr; // roll
    const y3 = x1 * sr + y2 * cr;
    xy.push([x3, -y3]); // the image's y axis points down
    depth.push(z2);
  }
  return { xy, depth };
}

export default function ThreeDPage({ params }: { params: Promise<{ person: string; session: string }> }) {
  const { person, session } = use(params);
  const [frame, setFrame] = useState(0);
  const [view, setView] = useState<View>("compensated");
  const [yaw, setYaw] = useState(0);
  const [pitch, setPitch] = useState(0);
  const [roll, setRoll] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState<(typeof PLAY_SPEEDS)[number]>(10);
  const drag = useRef<{ x: number; y: number } | null>(null);

  const pose = useQuery({ queryKey: ["headpose", person, session], queryFn: () => api.headPose(person, session) });
  const landmarks = useQuery({
    queryKey: ["face-landmarks", person, session, frame, view],
    queryFn: () => api.faceLandmarks(person, session, frame, view === "compensated"),
    enabled: pose.isSuccess,
    placeholderData: keepPreviousData,
  });

  const frames = pose.data?.time_s.length ?? 0;

  // Playback steps through the frames at the chosen rate, and stops at the last frame.
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
    }, 1000 / fps);
    return () => window.clearInterval(id);
  }, [playing, frames, fps]);

  const angleOption = useMemo<EChartsOption | null>(() => {
    if (!pose.data) return null;
    return linesOption(pose.data.time_s, [
      ["yaw", pose.data.yaw_deg],
      ["pitch", pose.data.pitch_deg],
      ["roll", pose.data.roll_deg],
    ], "degrees");
  }, [pose.data]);

  const depthOption = useMemo<EChartsOption | null>(() => {
    if (!pose.data) return null;
    return linesOption(pose.data.time_s, [["depth span", pose.data.depth_range]], "relative depth");
  }, [pose.data]);

  const projection = useMemo<EChartsOption | null>(() => {
    const points = (landmarks.data?.points ?? []) as Point[];
    const { xy, depth } = rotatedPoints(points, yaw, pitch, roll);
    if (xy.length === 0) return null;
    return {
      grid: { left: 40, right: 80, top: 20, bottom: 30 },
      xAxis: { type: "value", show: false, scale: true },
      yAxis: { type: "value", show: false, scale: true },
      visualMap: {
        dimension: 2,
        min: Math.min(...depth),
        max: Math.max(...depth),
        orient: "vertical",
        right: 0,
        top: "center",
        text: ["near", "far"],
        calculable: false,
        inRange: { color: ["#1d4ed8", "#a3a3a3", "#dc2626"] },
      },
      series: [{
        type: "scatter",
        symbolSize: 3,
        data: xy.map(([x, y], i) => [x, y, depth[i]]),
      }],
    };
  }, [landmarks.data, yaw, pitch, roll]);

  const angleAt = (values: (number | null)[] | undefined) => {
    const v = values?.[frame];
    return v == null ? "n/a" : `${v.toFixed(1)}°`;
  };

  // Dragging the plot turns the face. The chosen turn stays while the frames change.
  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    drag.current = { x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    drag.current = { x: e.clientX, y: e.clientY };
    setYaw((v) => Math.max(-180, Math.min(180, v + dx * 0.5)));
    setPitch((v) => Math.max(-90, Math.min(90, v + dy * 0.5)));
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  return (
    <div className="flex flex-col gap-6">
      <header>
        <p className="text-sm text-[var(--muted)]">
          <Link href={`/people/${encodeURIComponent(person)}`} className="underline">{person}</Link> / {session}
        </p>
        <h1 className="text-2xl font-semibold">3D facial landmarks</h1>
        <p className="text-sm text-[var(--muted)]">
          Drag the plot, or use the sliders, to turn the face. The view stays where you put it while the frames change.
          It is a 3D projection drawn on a plane, not a full 3D model.
        </p>
      </header>

      {pose.isPending ? <Loading label="Loading head pose…" /> : null}
      {pose.isError ? <ErrorState message={pose.error.message} code={(pose.error as { code?: string }).code} /> : null}

      {pose.data && frames > 0 ? (
        <Card title="Frame">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div className="flex w-full max-w-xl flex-col gap-2">
              <label className="flex flex-col gap-1 text-sm">
                Frame: {frame} of {frames - 1}
                <input
                  type="range"
                  min={0}
                  max={frames - 1}
                  value={frame}
                  onChange={(e) => { setPlaying(false); setFrame(Number(e.target.value)); }}
                  aria-label="Frame"
                />
              </label>
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <button
                  type="button"
                  onClick={() => {
                    if (!playing && frame >= frames - 1) setFrame(0);
                    setPlaying(!playing);
                  }}
                  aria-pressed={playing}
                  className="rounded border border-[var(--border)] px-3 py-1"
                >
                  {playing ? "Pause" : "Play"}
                </button>
                <button type="button" disabled={playing || frame <= 0} onClick={() => setFrame(frame - 1)}
                  className="rounded border border-[var(--border)] px-2 py-1 disabled:opacity-40">Previous</button>
                <button type="button" disabled={playing || frame >= frames - 1} onClick={() => setFrame(frame + 1)}
                  className="rounded border border-[var(--border)] px-2 py-1 disabled:opacity-40">Next</button>
                <label className="flex items-center gap-1">
                  Speed
                  <select
                    value={fps}
                    onChange={(e) => setFps(Number(e.target.value) as (typeof PLAY_SPEEDS)[number])}
                    className="rounded border border-[var(--border)] bg-transparent px-1 py-0.5"
                  >
                    {PLAY_SPEEDS.map((s) => <option key={s} value={s}>{s} frames/s</option>)}
                  </select>
                </label>
              </div>
            </div>
            <fieldset className="flex gap-4 text-sm">
              <legend className="sr-only">Landmark view</legend>
              <label className="flex items-center gap-2">
                <input type="radio" name="view" checked={view === "compensated"} onChange={() => setView("compensated")} />
                Head-pose compensated
              </label>
              <label className="flex items-center gap-2">
                <input type="radio" name="view" checked={view === "raw"} onChange={() => setView("raw")} />
                Raw as extracted
              </label>
            </fieldset>
          </div>
          <dl className="mt-4 grid grid-cols-3 gap-4 text-sm">
            <div><dt className="text-[var(--muted)]">Yaw (head)</dt><dd className="text-xl font-semibold">{angleAt(pose.data.yaw_deg)}</dd></div>
            <div><dt className="text-[var(--muted)]">Pitch (head)</dt><dd className="text-xl font-semibold">{angleAt(pose.data.pitch_deg)}</dd></div>
            <div><dt className="text-[var(--muted)]">Roll (head)</dt><dd className="text-xl font-semibold">{angleAt(pose.data.roll_deg)}</dd></div>
          </dl>
        </Card>
      ) : null}

      {projection ? (
        <Card title={`Landmarks, frame ${frame} (${view === "compensated" ? "head-pose compensated" : "raw"})`}>
          <div className="mb-3 grid gap-3 text-sm md:grid-cols-4">
            <label className="flex flex-col gap-1">
              Turn left-right: {yaw.toFixed(0)}°
              <input type="range" min={-180} max={180} value={yaw} onChange={(e) => setYaw(Number(e.target.value))} />
            </label>
            <label className="flex flex-col gap-1">
              Tilt up-down: {pitch.toFixed(0)}°
              <input type="range" min={-90} max={90} value={pitch} onChange={(e) => setPitch(Number(e.target.value))} />
            </label>
            <label className="flex flex-col gap-1">
              Roll: {roll.toFixed(0)}°
              <input type="range" min={-180} max={180} value={roll} onChange={(e) => setRoll(Number(e.target.value))} />
            </label>
            <div className="flex items-end">
              <button type="button" onClick={() => { setYaw(0); setPitch(0); setRoll(0); }}
                className="rounded border border-[var(--border)] px-3 py-1">Reset view</button>
            </div>
          </div>
          <div
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            className="cursor-grab touch-none select-none active:cursor-grabbing"
          >
            <Chart option={projection} height={420} label={`Landmarks for frame ${frame}`} />
          </div>
          <p className="mt-2 text-xs text-[var(--muted)]">Drag to turn the face. Colour shows depth: red is nearer, blue is further.</p>
        </Card>
      ) : landmarks.isSuccess ? (
        <Card title="Landmarks">
          <p className="text-sm text-[var(--muted)]">No face was tracked in frame {frame}.</p>
        </Card>
      ) : null}

      {angleOption ? (
        <Card title="Head rotation over the session (degrees)">
          <Chart option={angleOption} label="Head rotation over the session" />
        </Card>
      ) : null}
      {depthOption ? (
        <Card title="Face depth span over the session">
          <Chart option={depthOption} height={220} label="Face depth span over the session" />
        </Card>
      ) : null}
    </div>
  );
}
