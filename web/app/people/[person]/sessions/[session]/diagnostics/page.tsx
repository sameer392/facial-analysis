"use client";

import { use, useMemo, useState } from "react";
import Link from "next/link";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import type { EChartsOption } from "echarts";
import { api } from "../../../../../lib/api";
import { Card, ErrorState, Loading } from "../../../../../components/ui";
import { Chart } from "../../../../../components/chart";

export default function DiagnosticsPage({ params }: { params: Promise<{ person: string; session: string }> }) {
  const { person, session } = use(params);
  const [frame, setFrame] = useState(0);
  const summary = useQuery({ queryKey: ["face-summary", person, session], queryFn: () => api.faceSummary(person, session) });
  const series = useQuery({
    queryKey: ["face-series", person, session, "diag"],
    queryFn: () => api.faceSeries(person, session, ["face_detected", "is_neutral", "neutral_offset"]),
  });
  const raw = useQuery({
    queryKey: ["face-landmarks", person, session, frame, "raw"],
    placeholderData: keepPreviousData,
    queryFn: () => api.faceLandmarks(person, session, frame, false),
    enabled: summary.isSuccess,
  });
  const compensated = useQuery({
    queryKey: ["face-landmarks", person, session, frame, "comp"],
    placeholderData: keepPreviousData,
    queryFn: () => api.faceLandmarks(person, session, frame, true),
    enabled: summary.isSuccess,
  });

  const trackingOption = useMemo<EChartsOption | null>(() => {
    if (!series.data) return null;
    const t = series.data.time_s;
    return {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "time (s)" },
      yAxis: { type: "value", min: 0, max: 1.05 },
      series: [
        { name: "face detected", type: "line", step: "end", showSymbol: false, data: t.map((x, i) => [x, series.data.face_detected[i] ? 1 : 0]) },
        { name: "neutral frame", type: "line", step: "end", showSymbol: false, data: t.map((x, i) => [x, series.data.is_neutral[i] ? 1 : 0]) },
      ],
    };
  }, [series.data]);

  const landmarkOption = (points: (number | null)[][] | undefined, title: string): EChartsOption => ({
    title: { text: title, left: "center", textStyle: { fontSize: 13 } },
    grid: { left: 40, right: 20, top: 40, bottom: 30 },
    xAxis: { type: "value", show: false },
    yAxis: { type: "value", show: false, inverse: true },
    series: [
      {
        type: "scatter",
        symbolSize: 3,
        data: (points ?? []).filter((p) => p[0] != null && p[1] != null).map((p) => [p[0], p[1]]),
      },
    ],
  });

  const frames = summary.data?.frames ?? 1;
  const q = summary.data?.quality;

  return (
    <div className="flex flex-col gap-6">
      <header>
        <p className="text-sm text-[var(--muted)]">
          <Link href={`/people/${encodeURIComponent(person)}`} className="underline">{person}</Link> / {session}
        </p>
        <h1 className="text-2xl font-semibold">Diagnostics</h1>
        <p className="text-sm text-[var(--muted)]">Face tracking, neutral frames and landmarks. Tracking checks only.</p>
      </header>

      {summary.isPending ? <Loading label="Loading summary…" /> : null}
      {summary.isError ? <ErrorState message={summary.error.message} code={(summary.error as { code?: string }).code} /> : null}

      {q ? (
        <Card title="Tracking checks">
          <dl className="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
            <div><dt className="text-[var(--muted)]">Interruptions</dt><dd className="text-xl font-semibold">{q.interruption_count}</dd></div>
            <div><dt className="text-[var(--muted)]">Matrix available</dt><dd className="text-xl font-semibold">{q.transform_matrix_on_detected == null ? "n/a" : `${(q.transform_matrix_on_detected * 100).toFixed(1)} %`}</dd></div>
            <div><dt className="text-[var(--muted)]">Landmarks complete</dt><dd className="text-xl font-semibold">{q.landmark_completeness_on_detected == null ? "n/a" : `${(q.landmark_completeness_on_detected * 100).toFixed(1)} %`}</dd></div>
            <div><dt className="text-[var(--muted)]">Timestamp issues</dt><dd className="text-xl font-semibold">{q.timestamp_issues.length === 0 ? "none" : q.timestamp_issues.length}</dd></div>
          </dl>
          {q.interruptions.length > 0 ? (
            <ul className="mt-3 list-disc pl-5 text-sm">
              {q.interruptions.map((r) => (
                <li key={r.start_frame}>frames {r.start_frame} to {r.end_frame} ({r.length_frames} frames without a face)</li>
              ))}
            </ul>
          ) : null}
        </Card>
      ) : null}

      {trackingOption ? (
        <Card title="Face detection and neutral frames over time">
          <Chart option={trackingOption} height={200} label="Face detection and neutral frames over time" />
        </Card>
      ) : null}

      <Card title="Landmarks for one frame">
        <label className="mb-3 flex max-w-md flex-col gap-1 text-sm">
          Frame: {frame}
          <input
            type="range"
            min={0}
            max={Math.max(frames - 1, 0)}
            value={frame}
            onChange={(e) => setFrame(Number(e.target.value))}
            aria-label="Frame"
          />
        </label>
        <div className="grid gap-4 md:grid-cols-2">
          {raw.isPending ? <Loading label="Loading raw landmarks…" /> : <Chart option={landmarkOption(raw.data?.points, "Raw (normalised image units)")} label="Raw landmarks" />}
          {compensated.isPending ? <Loading label="Loading compensated landmarks…" /> : <Chart option={landmarkOption(compensated.data?.points, "Head-pose compensated (face-size units)")} label="Compensated landmarks" />}
        </div>
        <p className="mt-2 text-xs text-[var(--muted)]">Frames without a face have no points.</p>
      </Card>
    </div>
  );
}
