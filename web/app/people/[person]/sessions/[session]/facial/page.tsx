"use client";

import { use, useMemo } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import type { EChartsOption } from "echarts";
import { api } from "../../../../../lib/api";
import { Card, ErrorState, Loading } from "../../../../../components/ui";
import { Chart } from "../../../../../components/chart";

const FIELDS = ["velocity", "acceleration", "displacement", "neutral_offset", "jaw_open", "mouth_activity", "eye_blink", "yaw_deg", "pitch_deg", "roll_deg"];

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

export default function FacialPage({ params }: { params: Promise<{ person: string; session: string }> }) {
  const { person, session } = use(params);
  const summary = useQuery({ queryKey: ["face-summary", person, session], queryFn: () => api.faceSummary(person, session) });
  const series = useQuery({ queryKey: ["face-series", person, session], queryFn: () => api.faceSeries(person, session, FIELDS) });
  const blinks = useQuery({ queryKey: ["face-blinks", person, session], queryFn: () => api.faceBlinks(person, session) });

  const motionOption = useMemo(() => {
    if (!series.data) return null;
    const t = series.data.time_s;
    return lineOption(t, [
      ["velocity", series.data.velocity],
      ["acceleration", series.data.acceleration],
      ["displacement", series.data.displacement],
      ["neutral offset", series.data.neutral_offset],
    ]);
  }, [series.data]);

  const expressionOption = useMemo(() => {
    if (!series.data) return null;
    return lineOption(series.data.time_s, [
      ["jaw open", series.data.jaw_open],
      ["mouth activity", series.data.mouth_activity],
      ["eye blink", series.data.eye_blink],
    ]);
  }, [series.data]);

  const headOption = useMemo(() => {
    if (!series.data) return null;
    return lineOption(series.data.time_s, [
      ["yaw", series.data.yaw_deg],
      ["pitch", series.data.pitch_deg],
      ["roll", series.data.roll_deg],
    ]);
  }, [series.data]);

  const q = summary.data?.quality;
  const pct = (v: number | null | undefined) => (v == null ? "n/a" : `${(v * 100).toFixed(1)} %`);

  return (
    <div className="flex flex-col gap-6">
      <header>
        <p className="text-sm text-[var(--muted)]">
          <Link href={`/people/${encodeURIComponent(person)}`} className="underline">{person}</Link> / {session}
        </p>
        <h1 className="text-2xl font-semibold">Facial analysis</h1>
        <p className="text-sm text-[var(--muted)]">
          Tracking figures and motion values from the face analysis. Tracking figures do not measure motion accuracy.
        </p>
      </header>

      {summary.isPending ? <Loading label="Loading summary…" /> : null}
      {summary.isError ? <ErrorState message={summary.error.message} code={(summary.error as { code?: string }).code} /> : null}
      {q ? (
        <Card title="Tracking quality">
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
        </Card>
      ) : null}

      {series.isPending ? <Loading label="Loading series…" /> : null}
      {series.isError ? <ErrorState message={series.error.message} /> : null}
      {motionOption ? (
        <Card title="Motion over time (velocity, acceleration, displacement, neutral offset)">
          <Chart option={motionOption} label="Motion over time" />
        </Card>
      ) : null}
      {expressionOption ? (
        <Card title="Mouth and eye blendshapes">
          <Chart option={expressionOption} label="Mouth and eye blendshapes over time" />
        </Card>
      ) : null}
      {headOption ? (
        <Card title="Head rotation (degrees, from the transformation matrix)">
          <Chart option={headOption} label="Head rotation over time" />
        </Card>
      ) : null}

      <Card title="Blink events">
        {blinks.isPending ? <Loading /> : null}
        {blinks.data && blinks.data.length === 0 ? <p className="text-sm text-[var(--muted)]">No blinks detected at the 0.5 threshold.</p> : null}
        {blinks.data && blinks.data.length > 0 ? (
          <table className="w-full text-left text-sm">
            <thead className="text-[var(--muted)]">
              <tr><th scope="col" className="py-2 font-medium">Start frame</th><th scope="col" className="py-2 font-medium">End frame</th><th scope="col" className="py-2 font-medium">Start (s)</th><th scope="col" className="py-2 font-medium">Duration (ms)</th></tr>
            </thead>
            <tbody>
              {blinks.data.map((b) => (
                <tr key={`${b.start_frame}-${b.end_frame}`} className="border-t border-[var(--border)]">
                  <td className="py-2">{b.start_frame}</td><td className="py-2">{b.end_frame}</td>
                  <td className="py-2">{b.start_s?.toFixed(3) ?? "n/a"}</td><td className="py-2">{b.duration_ms?.toFixed(0) ?? "n/a"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </Card>
    </div>
  );
}
