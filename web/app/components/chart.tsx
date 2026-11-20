"use client";

import { useEffect, useRef } from "react";
import * as echarts from "echarts";

/**
 * ECharts wrapper. The chart is created once and updated in place when the option changes, so playback and
 * slider moves do not blank the chart. It is disposed only when the component unmounts.
 */
export function Chart({ option, height = 280, label }: { option: echarts.EChartsOption; height?: number; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chartRef.current = chart;
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    chartRef.current?.setOption(option, { notMerge: true });
  }, [option]);

  return <div ref={ref} role="img" aria-label={label} style={{ width: "100%", height }} />;
}
