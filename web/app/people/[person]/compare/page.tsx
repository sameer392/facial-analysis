"use client";

import { use, useMemo, useState } from "react";
import Link from "next/link";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import type { EChartsOption } from "echarts";
import { api } from "../../../lib/api";
import { Card, Empty, ErrorState, Loading } from "../../../components/ui";
import { Chart } from "../../../components/chart";

const METRIC_LABEL: Record<string, string> = {
  jaw_open: "Jaw opening",
  mouth_activity: "Mouth activity",
  eye_blink: "Eye blink score",
  brow_inner_up: "Eyebrow (inner up)",
  velocity: "Velocity",
  acceleration: "Acceleration",
};

export default function ComparePage({ params }: { params: Promise<{ person: string }> }) {
  const { person } = use(params);
  const sessions = useQuery({ queryKey: ["sessions", person], queryFn: () => api.sessions(person) });
  const faceSessions = (sessions.data ?? []).filter((s) => s.layout === "face" || s.layout === "pose").map((s) => s.session_id);
  const [picked, setPicked] = useState<string[] | null>(null);
  const chosen = picked ?? faceSessions;
  const comparison = useQuery({
    queryKey: ["comparison", person, chosen.join(",")],
    placeholderData: keepPreviousData,
    queryFn: () => api.comparison(person, chosen),
    enabled: chosen.length > 0,
  });

  const charts = useMemo(() => {
    if (!comparison.data) return [];
    return Object.entries(comparison.data.distributions).map(([metric, hist]) => {
      const sessionIds = Object.keys(hist.shares);
      const categories = hist.edges.slice(0, -1).map((e, i) =>
        e == null || hist.edges[i + 1] == null ? "" : `${(e as number).toFixed(2)}–${(hist.edges[i + 1] as number).toFixed(2)}`,
      );
      const option: EChartsOption = {
        tooltip: { trigger: "axis" },
        legend: { top: 0 },
        grid: { left: 50, right: 20, top: 40, bottom: 60 },
        xAxis: { type: "category", data: categories, axisLabel: { rotate: 45, fontSize: 9 } },
        yAxis: { type: "value", name: "share of frames" },
        series: sessionIds.map((sid) => ({
          name: sid,
          type: "bar",
          barGap: "-100%",
          data: hist.shares[sid].map((v) => v ?? 0),
        })),
      };
      return { metric, option };
    });
  }, [comparison.data]);

  const toggle = (id: string) => {
    setPicked((current) => {
      const base = current ?? faceSessions;
      return base.includes(id) ? base.filter((s) => s !== id) : [...base, id];
    });
  };

  return (
    <div className="flex flex-col gap-6">
      <header>
        <p className="text-sm text-[var(--muted)]"><Link href={`/people/${encodeURIComponent(person)}`} className="underline">{person}</Link> / compare</p>
        <h1 className="text-2xl font-semibold">Dataset comparison</h1>
        <p className="text-sm text-[var(--muted)]">Compare the distributions of several sessions. Shares of frames, on a shared range.</p>
      </header>

      {sessions.isPending ? <Loading label="Loading sessions…" /> : null}
      {sessions.isError ? <ErrorState message={sessions.error.message} /> : null}
      {sessions.data && faceSessions.length === 0 ? <Empty title="No analysed sessions" detail="Only sessions with face data can be compared." /> : null}

      <Card title="Sessions to compare">
        <fieldset className="flex flex-wrap gap-4 text-sm">
          <legend className="sr-only">Sessions</legend>
          {faceSessions.map((id) => (
            <label key={id} className="flex items-center gap-2">
              <input type="checkbox" checked={chosen.includes(id)} onChange={() => toggle(id)} />
              <span className="font-mono">{id}</span>
            </label>
          ))}
        </fieldset>
      </Card>

      {comparison.isPending && chosen.length > 0 ? <Loading label="Computing comparison…" /> : null}
      {comparison.isError ? <ErrorState message={comparison.error.message} code={(comparison.error as { code?: string }).code} /> : null}

      {comparison.data ? (
        <Card title="Session summary">
          <table className="w-full text-left text-sm">
            <thead className="text-[var(--muted)]">
              <tr>
                <th scope="col" className="py-2 font-medium">Session</th>
                <th scope="col" className="py-2 font-medium">Frames</th>
                <th scope="col" className="py-2 font-medium">Quality score</th>
                <th scope="col" className="py-2 font-medium">Detection</th>
                <th scope="col" className="py-2 font-medium">Blinks</th>
                <th scope="col" className="py-2 font-medium">Neutral frames</th>
                <th scope="col" className="py-2 font-medium">Landmark stability</th>
              </tr>
            </thead>
            <tbody>
              {comparison.data.sessions.map((r) => (
                <tr key={r.session_id} className="border-t border-[var(--border)]">
                  <td className="py-2 font-mono">{r.session_id}</td>
                  <td className="py-2">{r.frames}</td>
                  <td className="py-2">{r.quality_score ?? "n/a"}</td>
                  <td className="py-2">{r.detection_rate == null ? "n/a" : `${(r.detection_rate * 100).toFixed(1)} %`}</td>
                  <td className="py-2">{r.blink_events}</td>
                  <td className="py-2">{r.neutral_frames}</td>
                  <td className="py-2">{r.landmark_stability == null ? "n/a" : r.landmark_stability.toFixed(5)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-[var(--muted)]">
            Landmark stability is the median per-landmark standard deviation of head-pose compensated positions. Lower values mean the face shape was steadier.
          </p>
        </Card>
      ) : null}

      {comparison.data && Object.keys(comparison.data.blendshape_means).length > 0 ? (
        <Card title="Blendshape statistics (mean score per session)">
          <table className="w-full text-left text-sm">
            <thead className="text-[var(--muted)]">
              <tr>
                <th scope="col" className="py-2 font-medium">Blendshape</th>
                {comparison.data.sessions.map((r) => (
                  <th key={r.session_id} scope="col" className="py-2 font-medium font-mono">{r.session_id}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(comparison.data.blendshape_means).map(([name, bySession]) => (
                <tr key={name} className="border-t border-[var(--border)]">
                  <td className="py-2">{name}</td>
                  {comparison.data.sessions.map((r) => (
                    <td key={r.session_id} className="py-2">{bySession[r.session_id] == null ? "n/a" : bySession[r.session_id]!.toFixed(4)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ) : null}

      {charts.map(({ metric, option }) => (
        <Card key={metric} title={METRIC_LABEL[metric] ?? metric}>
          <Chart option={option} height={260} label={`${METRIC_LABEL[metric] ?? metric} distribution`} />
        </Card>
      ))}
    </div>
  );
}
