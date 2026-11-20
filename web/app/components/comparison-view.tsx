"use client";

import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { Comparison } from "../lib/api";
import { Card } from "./ui";
import { Chart } from "./chart";

const METRIC_LABEL: Record<string, string> = {
  jaw_open: "Jaw opening",
  mouth_activity: "Mouth activity",
  eye_blink: "Eye blink score",
  brow_inner_up: "Eyebrow (inner up)",
  velocity: "Velocity",
  acceleration: "Acceleration",
};

/** The comparison tables and distribution charts. Labels are session ids, or person/session across people. */
export function ComparisonView({ data }: { data: Comparison }) {
  const charts = useMemo(() => {
    return Object.entries(data.distributions).map(([metric, hist]) => {
      const sessionIds = Object.keys(hist.shares);
      const categories = hist.edges.slice(0, -1).map((e, i) =>
        e == null || hist.edges[i + 1] == null ? "" : `${(e as number).toFixed(2)}–${(hist.edges[i + 1] as number).toFixed(2)}`,
      );
      const option: EChartsOption = {
        tooltip: { trigger: "axis" },
        legend: { top: 0, type: "scroll" },
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
  }, [data]);

  return (
    <>
      <Card title="Session summary">
        <div className="overflow-x-auto">
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
              {data.sessions.map((r) => (
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
        </div>
        <p className="mt-2 text-xs text-[var(--muted)]">
          Landmark stability is the median per-landmark standard deviation of head-pose compensated positions. Lower values mean the face shape was steadier.
        </p>
      </Card>

      {Object.keys(data.blendshape_means).length > 0 ? (
        <Card title="Blendshape statistics (mean score per session)">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-[var(--muted)]">
                <tr>
                  <th scope="col" className="py-2 font-medium">Blendshape</th>
                  {data.sessions.map((r) => (
                    <th key={r.session_id} scope="col" className="py-2 font-medium font-mono">{r.session_id}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.blendshape_means).map(([name, bySession]) => (
                  <tr key={name} className="border-t border-[var(--border)]">
                    <td className="py-2">{name}</td>
                    {data.sessions.map((r) => (
                      <td key={r.session_id} className="py-2">{bySession[r.session_id] == null ? "n/a" : bySession[r.session_id]!.toFixed(4)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : null}

      {charts.map(({ metric, option }) => (
        <Card key={metric} title={METRIC_LABEL[metric] ?? metric}>
          <Chart option={option} height={260} label={`${METRIC_LABEL[metric] ?? metric} distribution`} />
        </Card>
      ))}
    </>
  );
}
