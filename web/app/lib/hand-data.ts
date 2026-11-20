/** Helpers for the Hands page: pipeline stages, bounded paging, and the MediaPipe hand topology. */

import type { HandTablePage, Job } from "./api";

export const HAND_STAGES: { key: string; label: string }[] = [
  { key: "hand landmarks", label: "Hand landmarks" },
  { key: "hand tracking", label: "Tracking and motion" },
  { key: "gestures", label: "Gesture classification" },
  { key: "validation", label: "Dataset validation" },
];

export type StageState = "done" | "running" | "failed" | "cancelled" | "not run" | "pending";

/**
 * The state of each pipeline stage for one job. A stage is "done" only when the job succeeded or a later stage
 * started. A failed job marks its stage failed and the stages after it "not run", so a partial run never looks
 * like a complete one.
 */
export function stageStates(job: Job | null): StageState[] {
  if (!job) return HAND_STAGES.map(() => "pending");
  const current = Math.max(0, HAND_STAGES.findIndex((s) => s.key === job.stage));
  return HAND_STAGES.map((_, i) => {
    if (job.state === "succeeded") return "done";
    if (job.state === "queued") return "pending";
    if (job.state === "running") {
      if (i < current) return "done";
      if (i === current) return "running";
      return "pending";
    }
    // failed, cancelled or interrupted
    if (i < current) return "done";
    if (i === current) return job.state === "cancelled" ? "cancelled" : "failed";
    return "not run";
  });
}

/** Rows per request for the table endpoints (the API maximum). */
export const PAGE_ROWS = 200;
/** Most rows a chart or filter loads in one go. Longer sessions show the first rows and say so. */
export const MAX_LOADED_ROWS = 5000;

/**
 * Loads rows page by page, from the first row, up to MAX_LOADED_ROWS. Each request is one page, so a long
 * recording does not load in one response.
 */
export async function loadRows<T>(
  fetchPage: (offset: number, limit: number) => Promise<HandTablePage<T>>,
): Promise<{ rows: T[]; total: number; truncated: boolean }> {
  const first = await fetchPage(0, PAGE_ROWS);
  const rows = [...first.rows];
  const total = first.total;
  let offset = rows.length;
  while (offset < total && rows.length < MAX_LOADED_ROWS) {
    const page = await fetchPage(offset, PAGE_ROWS);
    if (page.rows.length === 0) break;
    rows.push(...page.rows);
    offset += page.rows.length;
  }
  return { rows, total, truncated: rows.length < total };
}

/** Frame time in milliseconds, using the same rule as the H1 timeline: round(frame * 1000 / fps). */
export function frameTimeMs(frame: number, fps: number | null): number | null {
  if (!fps || fps <= 0) return null;
  return Math.round((frame * 1000) / fps);
}

export const HAND_LANDMARK_NAMES = [
  "wrist",
  "thumb CMC", "thumb MCP", "thumb IP", "thumb tip",
  "index MCP", "index PIP", "index DIP", "index tip",
  "middle MCP", "middle PIP", "middle DIP", "middle tip",
  "ring MCP", "ring PIP", "ring DIP", "ring tip",
  "pinky MCP", "pinky PIP", "pinky DIP", "pinky tip",
] as const;

/** Landmark index pairs joined by the MediaPipe hand model (finger chains and the palm). */
export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export const FINGERTIP_INDICES = [4, 8, 12, 16, 20] as const;

/** Format a number for display, or "n/a" when it is missing. */
export function fmt(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "n/a" : value.toFixed(digits);
}
