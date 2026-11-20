"use client";

import "echarts-gl";
import { use, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { EChartsOption } from "echarts";
import { api, type FaceTopology } from "../../../../../lib/api";
import { Card, ErrorState, Loading } from "../../../../../components/ui";
import { Chart } from "../../../../../components/chart";

type View = "compensated" | "raw";
type Mode = "off" | "points" | "wireframe" | "contours" | "hybrid";
type Point = [number | null, number | null, number | null];
type Xyz = { x: number; y: number; z: number };

const PLAY_SPEEDS = [5, 10, 30] as const;
const MODES: { value: Mode; label: string }[] = [
  { value: "points", label: "Points" },
  { value: "wireframe", label: "Wireframe" },
  { value: "contours", label: "Contours" },
  { value: "hybrid", label: "Points and contours" },
  { value: "off", label: "Nothing" },
];
const ZOOM_MIN = 0.4;
const ZOOM_MAX = 6;
const BASE_DISTANCE = 220;

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
 * Turns the face by the chosen yaw, pitch and roll (the face itself, not the camera). Returns one entry per
 * landmark index, so the mesh edges can be drawn between the same points.
 */
function turnFace(points: Point[], yawDeg: number, pitchDeg: number, rollDeg: number) {
  const valid = points.filter((p): p is [number, number, number] => p[0] != null && p[1] != null && p[2] != null);
  const cx = valid.length ? valid.reduce((a, p) => a + p[0], 0) / valid.length : 0;
  const cy = valid.length ? valid.reduce((a, p) => a + p[1], 0) / valid.length : 0;
  const cz = valid.length ? valid.reduce((a, p) => a + p[2], 0) / valid.length : 0;
  const yaw = (yawDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;
  const roll = (rollDeg * Math.PI) / 180;
  const cyw = Math.cos(yaw);
  const syw = Math.sin(yaw);
  const cpt = Math.cos(pitch);
  const spt = Math.sin(pitch);
  const cr = Math.cos(roll);
  const sr = Math.sin(roll);
  let extent = 0;
  const out: (Xyz | null)[] = points.map((p) => {
    if (p[0] == null || p[1] == null || p[2] == null) return null;
    const x = p[0] - cx;
    const y = p[1] - cy;
    const z = p[2] - cz;
    const x1 = x * cyw + z * syw;
    const z1 = -x * syw + z * cyw;
    const y2 = y * cpt - z1 * spt;
    const z2 = y * spt + z1 * cpt;
    const x3 = x1 * cr - y2 * sr;
    const y3 = x1 * sr + y2 * cr;
    extent = Math.max(extent, Math.abs(x3), Math.abs(y3), Math.abs(z2));
    return { x: x3, y: -y3, z: z2 }; // the image's y axis points down
  });
  return { out, extent };
}

export default function ThreeDPage({ params }: { params: Promise<{ person: string; session: string }> }) {
  const { person, session } = use(params);
  const [frame, setFrame] = useState(0);
  const [view, setView] = useState<View>("compensated");
  const [mode, setMode] = useState<Mode>("points");
  const [yaw, setYaw] = useState(0);
  const [pitch, setPitch] = useState(0);
  const [roll, setRoll] = useState(0);
  const [camAlpha, setCamAlpha] = useState(15);
  const [camBeta, setCamBeta] = useState(-20);
  const [zoom, setZoom] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState<(typeof PLAY_SPEEDS)[number]>(10);
  const [plotEl, setPlotEl] = useState<HTMLDivElement | null>(null);

  const pose = useQuery({ queryKey: ["headpose", person, session], queryFn: () => api.headPose(person, session) });
  const landmarks = useQuery({
    queryKey: ["face-landmarks", person, session, frame, view],
    queryFn: () => api.faceLandmarks(person, session, frame, view === "compensated"),
    enabled: pose.isSuccess,
    placeholderData: keepPreviousData,
  });
  const topology = useQuery<FaceTopology>({ queryKey: ["face-topology"], queryFn: api.faceTopology, staleTime: Infinity });

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

  // Mouse: drag to turn the camera around the face; the wheel zooms. Native listeners, so the wheel does not scroll the page.
  useEffect(() => {
    const el = plotEl;
    if (!el) return;
    let last: { x: number; y: number } | null = null;
    const down = (e: PointerEvent) => {
      last = { x: e.clientX, y: e.clientY };
      el.setPointerCapture(e.pointerId);
    };
    const move = (e: PointerEvent) => {
      if (!last) return;
      const dx = e.clientX - last.x;
      const dy = e.clientY - last.y;
      last = { x: e.clientX, y: e.clientY };
      setCamBeta((v) => Math.max(-180, Math.min(180, v + dx * 0.5)));
      setCamAlpha((v) => Math.max(-90, Math.min(90, v + dy * 0.5)));
    };
    const up = () => {
      last = null;
    };
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      setZoom((z) => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, z * Math.exp(-e.deltaY * 0.0015))));
    };
    el.addEventListener("pointerdown", down);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("wheel", wheel, { passive: false });
    return () => {
      el.removeEventListener("pointerdown", down);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", up);
      el.removeEventListener("wheel", wheel);
    };
  }, [plotEl]);

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

  const option = useMemo<EChartsOption | null>(() => {
    const points = (landmarks.data?.points ?? []) as Point[];
    const { out, extent } = turnFace(points, yaw, pitch, roll);
    if (!out.some((p) => p !== null)) return null;
    const bound = (extent || 1) * 1.1;
    const depths = out.filter((p): p is Xyz => p !== null).map((p) => p.z);
    const showPoints = mode === "points" || mode === "hybrid";

    // Each mesh edge is drawn as a row of small 3D points between its two landmarks. The points are placed in 3D,
    // so they turn with the face. Edges with a missing end are skipped.
    const STEPS = 6;
    const edgeDots = (edges: number[][], colour: string, opacity: number, size: number) => ({
      type: "scatter3D",
      silent: true,
      symbolSize: size,
      itemStyle: { color: colour, opacity },
      data: edges.flatMap(([a, b]) => {
        const pa = out[a];
        const pb = out[b];
        if (!pa || !pb) return [];
        const dots: number[][] = [];
        for (let s = 0; s <= STEPS; s++) {
          const t = s / STEPS;
          dots.push([pa.x + (pb.x - pa.x) * t, pa.y + (pb.y - pa.y) * t, pa.z + (pb.z - pa.z) * t]);
        }
        return dots;
      }),
    });

    const series: Record<string, unknown>[] = [];
    if (mode === "wireframe" && topology.data) series.push(edgeDots(topology.data.mesh.edges, topology.data.mesh.colour, 0.7, 1.5));
    if (mode === "hybrid" && topology.data) series.push(edgeDots(topology.data.mesh.edges, topology.data.mesh.colour, 0.25, 1));
    if ((mode === "contours" || mode === "hybrid") && topology.data) {
      for (const region of Object.values(topology.data.regions)) {
        series.push(edgeDots(region.edges, region.colour, 0.95, 2));
      }
    }
    if (showPoints) {
      series.push({
        type: "scatter3D",
        symbolSize: 3,
        data: out.flatMap((p) => (p ? [[p.x, p.y, p.z]] : [])),
      });
    }

    // No axes, tick labels or box: only the face is shown.
    const axis = {
      type: "value",
      min: -bound,
      max: bound,
      show: false,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { show: false },
      splitLine: { show: false },
      axisPointer: { show: false },
    };
    const spec = {
      animation: false,
      grid3D: {
        show: false,
        boxWidth: 100,
        boxHeight: 100,
        boxDepth: 100,
        viewControl: {
          projection: "perspective",
          alpha: camAlpha,
          beta: camBeta,
          distance: BASE_DISTANCE / zoom,
          // The camera is driven by the controls on this page, so the built-in mouse handling is turned off.
          rotateSensitivity: 0,
          zoomSensitivity: 0,
          panSensitivity: 0,
          autoRotate: false,
        },
        light: { main: { intensity: 1.1, shadow: false }, ambient: { intensity: 0.35 } },
      },
      xAxis3D: axis,
      yAxis3D: axis,
      zAxis3D: axis,
      visualMap: showPoints
        ? {
            show: false,
            dimension: 2,
            seriesIndex: series.length - 1,
            min: Math.min(...depths),
            max: Math.max(...depths),
            inRange: { color: ["#1d4ed8", "#a3a3a3", "#dc2626"] },
          }
        : undefined,
      series,
    };
    return spec as unknown as EChartsOption;
  }, [landmarks.data, yaw, pitch, roll, camAlpha, camBeta, zoom, mode, topology.data]);

  const angleAt = (values: (number | null)[] | undefined) => {
    const v = values?.[frame];
    return v == null ? "n/a" : `${v.toFixed(1)}°`;
  };

  return (
    <div className="flex flex-col gap-6">
      <header>
        <p className="text-sm text-[var(--muted)]">
          <Link href={`/people/${encodeURIComponent(person)}`} className="underline">{person}</Link> / {session}
        </p>
        <h1 className="text-2xl font-semibold">3D facial landmarks</h1>
        <p className="text-sm text-[var(--muted)]">
          Drag the picture to turn it in 3D, and use the wheel or the + and − buttons to zoom. The view stays where you put
          it while the frames change.
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
            <fieldset className="flex flex-col gap-2 text-sm">
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

      {landmarks.data ? (
        <Card title={`Landmarks, frame ${frame} (${view === "compensated" ? "head-pose compensated" : "raw"})`}>
          <div className="mb-3 grid gap-3 text-sm md:grid-cols-6">
            <label className="flex flex-col gap-1">
              Show
              <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}
                className="rounded border border-[var(--border)] bg-transparent px-2 py-1">
                {MODES.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              Face turn (left-right): {yaw.toFixed(0)}°
              <input type="range" min={-180} max={180} value={yaw} onChange={(e) => setYaw(Number(e.target.value))} />
            </label>
            <label className="flex flex-col gap-1">
              Face tilt (up-down): {pitch.toFixed(0)}°
              <input type="range" min={-90} max={90} value={pitch} onChange={(e) => setPitch(Number(e.target.value))} />
            </label>
            <label className="flex flex-col gap-1">
              Face roll: {roll.toFixed(0)}°
              <input type="range" min={-180} max={180} value={roll} onChange={(e) => setRoll(Number(e.target.value))} />
            </label>
            <div className="flex items-end gap-2">
              <button type="button" aria-label="Zoom in" onClick={() => setZoom((z) => Math.min(ZOOM_MAX, z * 1.25))}
                className="rounded border border-[var(--border)] px-3 py-1">+</button>
              <button type="button" aria-label="Zoom out" onClick={() => setZoom((z) => Math.max(ZOOM_MIN, z / 1.25))}
                className="rounded border border-[var(--border)] px-3 py-1">−</button>
            </div>
            <div className="flex items-end">
              <button
                type="button"
                onClick={() => { setYaw(0); setPitch(0); setRoll(0); setCamAlpha(15); setCamBeta(-20); setZoom(1); }}
                className="rounded border border-[var(--border)] px-3 py-1"
              >
                Reset view
              </button>
            </div>
          </div>
          {option ? (
            <div ref={setPlotEl} className="cursor-grab touch-none select-none active:cursor-grabbing">
              <Chart option={option} height={480} label={`3D landmarks for frame ${frame}`} />
            </div>
          ) : (
            <p className="text-sm text-[var(--muted)]">No face was tracked in frame {frame}.</p>
          )}
          <p className="mt-2 text-xs text-[var(--muted)]">
            Drag to turn ({camBeta.toFixed(0)}°, {camAlpha.toFixed(0)}°), wheel to zoom ({zoom.toFixed(2)}×). Points are coloured by
            depth: red is nearer, blue is further. The mesh and contours use MediaPipe&apos;s own connections, as on the Overlay page.
          </p>
          {landmarks.data.points.some((p) => p[0] != null) ? (
            <details className="mt-4">
              <summary className="cursor-pointer text-sm underline">Coordinates for frame {frame} ({landmarks.data.points.length} landmarks)</summary>
              <div className="mt-3 max-h-[480px] overflow-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-[var(--muted)]">
                    <tr>
                      <th scope="col" className="py-2 pr-3 font-medium">ID</th>
                      <th scope="col" className="py-2 pr-3 font-medium">x</th>
                      <th scope="col" className="py-2 pr-3 font-medium">y</th>
                      <th scope="col" className="py-2 font-medium">z</th>
                    </tr>
                  </thead>
                  <tbody>
                    {landmarks.data.points.map((p, i) => (
                      <tr key={i} className="border-t border-[var(--border)]">
                        <td className="py-1 pr-3">{i}</td>
                        <td className="py-1 pr-3">{p[0] == null ? "n/a" : p[0].toFixed(5)}</td>
                        <td className="py-1 pr-3">{p[1] == null ? "n/a" : p[1].toFixed(5)}</td>
                        <td className="py-1">{p[2] == null ? "n/a" : p[2].toFixed(5)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          ) : null}
        </Card>
      ) : landmarks.isPending && pose.isSuccess ? (
        <Card title="Landmarks">
          <Loading label="Loading landmarks…" />
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
