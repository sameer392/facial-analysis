"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { EChartsOption } from "echarts";
import { api, type HandMotionRow } from "../lib/api";
import { fmt, loadRows } from "../lib/hand-data";
import { Chart } from "./chart";
import { Card, ErrorState, Loading, Empty } from "./ui";

const FINGERS = ["thumb", "index", "middle", "ring", "pinky"] as const;
const FINGER_ANGLE_KEYS: Record<(typeof FINGERS)[number], [string, string]> = {
  thumb: ["thumb_mcp_angle_deg", "thumb_ip_angle_deg"],
  index: ["index_pip_angle_deg", "index_dip_angle_deg"],
  middle: ["middle_pip_angle_deg", "middle_dip_angle_deg"],
  ring: ["ring_pip_angle_deg", "ring_dip_angle_deg"],
  pinky: ["pinky_pip_angle_deg", "pinky_dip_angle_deg"],
};

/** Value of one motion column for one row, or null when it is missing (never zero). */
function value(row: HandMotionRow, key: string): number | null {
  const v = row[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** A timestamp-aware line chart. Missing values stay gaps (connectNulls is off). */
function lineOption(rows: HandMotionRow[], keys: { key: string; name: string }[], yName: string): EChartsOption {
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0, type: "scroll" },
    grid: { left: 60, right: 20, top: 40, bottom: 40 },
    xAxis: { type: "value", name: "time (s)" },
    yAxis: { type: "value", name: yName, scale: true },
    series: keys.map(({ key, name }) => ({
      name,
      type: "line",
      showSymbol: false,
      connectNulls: false,
      data: rows.map((r) => [r.timestamp_ms === null ? null : r.timestamp_ms / 1000, value(r, key)] as [number | null, number | null]),
    })),
  };
}

export function HandMotionCharts({ person, session }: { person: string; session: string }) {
  const motion = useQuery({
    queryKey: ["hand-motion-all", person, session],
    queryFn: () => loadRows((offset, limit) => api.handMotion(person, session, offset, limit)),
  });
  const [track, setTrack] = useState<number | "">("");
  const [finger, setFinger] = useState<(typeof FINGERS)[number]>("index");

  const rows = motion.data?.rows ?? [];
  const tracks = useMemo(() => [...new Set(rows.map((r) => r.track_id))].sort((a, b) => a - b), [rows]);
  const chosen = track === "" ? tracks[0] : track;
  const selected = useMemo(() => rows.filter((r) => r.track_id === chosen).sort((a, b) => a.frame - b.frame), [rows, chosen]);

  const trajectory = useMemo<EChartsOption>(() => ({
    tooltip: { trigger: "item" },
    grid: { left: 60, right: 20, top: 30, bottom: 50 },
    xAxis: { type: "value", name: "centroid x (image pixels)", scale: true },
    yAxis: { type: "value", name: "centroid y (image pixels, down)", inverse: true, scale: true },
    series: [{
      type: "line",
      showSymbol: true,
      symbolSize: 4,
      connectNulls: false,
      data: selected.map((r) => [value(r, "centroid_x_px"), value(r, "centroid_y_px")]),
    }],
  }), [selected]);

  const displacement = useMemo(() => lineOption(selected, [
    { key: "displacement_px", name: "centroid displacement" },
    { key: "wrist_displacement_px", name: "wrist displacement" },
  ], "pixels between observations"), [selected]);
  const speed = useMemo(() => lineOption(selected, [{ key: "velocity_px_per_s", name: "velocity" }], "pixels per second"), [selected]);
  const accel = useMemo(() => lineOption(selected, [{ key: "acceleration_px_per_s2", name: "acceleration" }], "pixels per second squared"), [selected]);
  const angles = useMemo(() => lineOption(selected, [
    { key: "direction_deg", name: "movement direction" },
    { key: "orientation_deg", name: "palm orientation" },
  ], "degrees (image plane)"), [selected]);
  const openness = useMemo(() => lineOption(selected, [{ key: "openness", name: "openness" }], "ratio (palm lengths)"), [selected]);
  const fingerAngles = useMemo(() => lineOption(selected, [
    { key: FINGER_ANGLE_KEYS[finger][0], name: `${finger} joint 1` },
    { key: FINGER_ANGLE_KEYS[finger][1], name: `${finger} joint 2` },
  ], "degrees (image plane)"), [selected, finger]);

  const velocities = selected.map((r) => value(r, "velocity_px_per_s")).filter((v): v is number => v !== null);
  const median = velocities.length ? [...velocities].sort((a, b) => a - b)[Math.floor(velocities.length / 2)] : null;

  if (motion.isPending) return <Card title="Motion"><Loading label="Loading motion observations…" /></Card>;
  if (motion.isError) return <Card title="Motion"><ErrorState message={motion.error.message} code={(motion.error as { code?: string }).code} /></Card>;
  if (rows.length === 0) return <Card title="Motion"><Empty title="No linked observations" detail="Hand tracking has not linked any hand in this session." /></Card>;

  return (
    <div className="flex flex-col gap-6">
      <Card title="Motion: choose a track">
        <div className="flex flex-wrap items-center gap-4 text-sm">
          <label className="flex items-center gap-2">
            Track
            <select
              value={chosen ?? ""}
              onChange={(e) => setTrack(Number(e.target.value))}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
            >
              {tracks.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
          <span className="text-[var(--muted)]">
            {selected.length} observations on this track
            {motion.data?.truncated ? `. Only the first ${rows.length} of ${motion.data.total} observations are loaded for charts.` : ""}
          </span>
        </div>
        <p className="mt-2 text-xs text-[var(--muted)]">
          Distances and speeds are in image pixels: normalised x times the frame width, and y times the frame height.
          They are not physical distances, and pixel speed is not physical hand speed. Gaps are missing values, not zero.
        </p>
        <dl className="mt-3 grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
          <div><dt className="text-[var(--muted)]">Median velocity</dt><dd>{fmt(median, 1)} px/s</dd></div>
          <div><dt className="text-[var(--muted)]">Velocity values</dt><dd>{velocities.length} of {selected.length}</dd></div>
          <div><dt className="text-[var(--muted)]">Partial views</dt><dd>{selected.filter((r) => r.partial_view === 1).length}</dd></div>
          <div><dt className="text-[var(--muted)]">After a gap (no velocity)</dt><dd>{selected.filter((r) => r.motion_reason === "after_gap").length}</dd></div>
        </dl>
      </Card>

      <Card title="Trajectory of the hand centroid (image pixels)">
        <Chart option={trajectory} height={320} label="Centroid trajectory in image pixels" />
      </Card>
      <Card title="Displacement (image pixels per frame step)">
        <Chart option={displacement} label="Displacement over time" />
      </Card>
      <Card title="Velocity (image pixels per second)">
        <Chart option={speed} label="Velocity over time" />
      </Card>
      <Card title="Acceleration (image pixels per second squared)">
        <Chart option={accel} label="Acceleration over time" />
      </Card>
      <Card title="Movement direction and palm orientation (degrees, image plane)">
        <p className="mb-2 text-xs text-[var(--muted)]">Direction: 0 is towards +x (right), 90 towards +y (down). Orientation is the wrist to middle-knuckle direction in the image plane only.</p>
        <Chart option={angles} label="Movement direction and palm orientation over time" />
      </Card>
      <Card title="Hand openness (ratio)">
        <p className="mb-2 text-xs text-[var(--muted)]">Mean wrist-to-fingertip distance divided by palm length. Dimensionless.</p>
        <Chart option={openness} label="Hand openness over time" />
      </Card>
      <Card title="Finger joint angles (degrees, image plane)">
        <label className="mb-2 flex items-center gap-2 text-sm">
          Finger
          <select value={finger} onChange={(e) => setFinger(e.target.value as (typeof FINGERS)[number])}
            className="rounded border border-[var(--border)] bg-transparent px-2 py-1">
            {FINGERS.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </label>
        <Chart option={fingerAngles} label={`${finger} joint angles over time`} />
      </Card>
    </div>
  );
}
