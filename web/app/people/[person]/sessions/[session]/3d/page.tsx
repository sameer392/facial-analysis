"use client";

import { use, useMemo, useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import type { EChartsOption } from "echarts";
import { api } from "../../../../../lib/api";
import { Card, ErrorState, Loading } from "../../../../../components/ui";
import { Chart } from "../../../../../components/chart";

type View = "compensated" | "raw";

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

export default function ThreeDPage({ params }: { params: Promise<{ person: string; session: string }> }) {
  const { person, session } = use(params);
  const [frame, setFrame] = useState(0);
  const [view, setView] = useState<View>("compensated");

  const pose = useQuery({ queryKey: ["headpose", person, session], queryFn: () => api.headPose(person, session) });
  const landmarks = useQuery({
    queryKey: ["v1lm", person, session, frame, view],
    queryFn: () => api.v1Landmarks(person, session, frame, view === "compensated"),
    enabled: pose.isSuccess,
  });

  const frames = pose.data?.time_s.length ?? 0;

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
    const points = (landmarks.data?.points ?? []).filter((p) => p[0] != null && p[1] != null && p[2] != null);
    if (points.length === 0) return null;
    const zs = points.map((p) => p[2] as number);
    return {
      grid: { left: 40, right: 80, top: 20, bottom: 30 },
      xAxis: { type: "value", show: false },
      yAxis: { type: "value", show: false, inverse: true },
      visualMap: {
        dimension: 2,
        min: Math.min(...zs),
        max: Math.max(...zs),
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
        data: points.map((p) => [p[0], p[1], p[2]]),
      }],
    };
  }, [landmarks.data]);

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
          Landmarks and head pose from the version 1 analysis. The frame view is a projection: horizontal and vertical
          position with depth as colour. It is not a rotatable 3D model.
        </p>
      </header>

      {pose.isPending ? <Loading label="Loading head pose…" /> : null}
      {pose.isError ? <ErrorState message={pose.error.message} code={(pose.error as { code?: string }).code} /> : null}

      {pose.data && frames > 0 ? (
        <Card title="Frame">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <label className="flex w-full max-w-md flex-col gap-1 text-sm">
              Frame: {frame} of {frames - 1}
              <input type="range" min={0} max={frames - 1} value={frame} onChange={(e) => setFrame(Number(e.target.value))} aria-label="Frame" />
            </label>
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
            <div><dt className="text-[var(--muted)]">Yaw</dt><dd className="text-xl font-semibold">{angleAt(pose.data.yaw_deg)}</dd></div>
            <div><dt className="text-[var(--muted)]">Pitch</dt><dd className="text-xl font-semibold">{angleAt(pose.data.pitch_deg)}</dd></div>
            <div><dt className="text-[var(--muted)]">Roll</dt><dd className="text-xl font-semibold">{angleAt(pose.data.roll_deg)}</dd></div>
          </dl>
        </Card>
      ) : null}

      {landmarks.isPending && pose.isSuccess ? <Loading label="Loading landmarks…" /> : null}
      {landmarks.isError ? <ErrorState message={landmarks.error.message} code={(landmarks.error as { code?: string }).code} /> : null}
      {landmarks.data && !projection ? (
        <Card title="Landmarks">
          <p className="text-sm text-[var(--muted)]">No face was tracked in frame {frame}.</p>
        </Card>
      ) : null}
      {projection ? (
        <Card title={`Landmark projection, frame ${frame} (${view === "compensated" ? "head-pose compensated" : "raw"})`}>
          <Chart option={projection} height={420} label={`Landmark projection for frame ${frame}`} />
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
