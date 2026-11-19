"use client";

import { useEffect, useRef } from "react";
import * as echarts from "echarts";

/** Minimal ECharts wrapper. The option is set on mount and on change; the chart is disposed on unmount. */
export function Chart({ option, height = 280, label }: { option: echarts.EChartsOption; height?: number; label: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chart.setOption(option, true);
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
    };
  }, [option]);

  return <div ref={ref} role="img" aria-label={label} style={{ width: "100%", height }} />;
}
