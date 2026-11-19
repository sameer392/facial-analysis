/** Typed client for the read-only Python API. Requests go to /api, which Next.js proxies to 127.0.0.1. */

export type Health = {
  status: string;
  api_version: string;
  schema_version: string;
  data_root: string;
  read_only: boolean;
};

export type SessionRow = {
  session_id: string;
  has_video: boolean;
  layout: "v1" | "v2" | "incomplete_v2" | "none";
};

export type Quality = {
  session_id: string;
  frames: number;
  detection_rate: number | null;
  transform_matrix_rate: number | null;
  longest_missing_run_frames: number;
  median_landmark_jitter: number | null;
  quality_score: number | null;
  landmark_completeness_on_detected: number | null;
  transform_matrix_on_detected: number | null;
  interruption_count: number;
  interruptions: { start_frame: number; end_frame: number; length_frames: number }[];
  timestamp_issues: string[];
  frame_size_known: boolean;
  scope: string;
};

export type V1Summary = {
  session_id: string;
  frames: number;
  neutral_frames: number;
  blink_events: number;
  quality: Quality;
};

export type Series = { time_s: (number | null)[] } & Record<string, (number | boolean | null)[]>;

export type Blink = {
  start_frame: number;
  end_frame: number;
  start_s: number | null;
  duration_ms: number | null;
  peak_score: number | null;
};

export type LandmarkFrame = { index: number; compensated: boolean; points: (number | null)[][] };

export type Comparison = {
  sessions: {
    session_id: string;
    frames: number;
    quality_score: number | null;
    detection_rate: number | null;
    blink_events: number;
    neutral_frames: number;
  }[];
  distributions: Record<string, { edges: (number | null)[]; shares: Record<string, (number | null)[]> }>;
};

export class ApiFailure extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`/api${path}`, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    let code = "http_error";
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { code?: string; message?: string };
      code = body.code ?? code;
      message = body.message ?? message;
    } catch {
      // not JSON: keep the generic message
    }
    throw new ApiFailure(response.status, code, message);
  }
  return (await response.json()) as T;
}

const enc = encodeURIComponent;

export const api = {
  health: () => get<Health>("/health"),
  persons: () => get<string[]>("/persons"),
  sessions: (person: string) => get<SessionRow[]>(`/persons/${enc(person)}/sessions`),
  v1Summary: (person: string, session: string) =>
    get<V1Summary>(`/persons/${enc(person)}/sessions/${enc(session)}/v1/summary`),
  v1Series: (person: string, session: string, fields: string[]) =>
    get<Series>(`/persons/${enc(person)}/sessions/${enc(session)}/v1/series?fields=${enc(fields.join(","))}`),
  v1Blinks: (person: string, session: string) =>
    get<Blink[]>(`/persons/${enc(person)}/sessions/${enc(session)}/v1/blinks`),
  v1Landmarks: (person: string, session: string, index: number, compensated: boolean) =>
    get<LandmarkFrame>(
      `/persons/${enc(person)}/sessions/${enc(session)}/v1/landmarks/${index}?compensated=${compensated}`,
    ),
  comparison: (person: string, sessions: string[]) =>
    get<Comparison>(`/persons/${enc(person)}/comparison?sessions=${enc(sessions.join(","))}`),
};
