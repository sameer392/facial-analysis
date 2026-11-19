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

export type HeadPose = {
  time_s: (number | null)[];
  yaw_deg: (number | null)[];
  pitch_deg: (number | null)[];
  roll_deg: (number | null)[];
  depth_range: (number | null)[];
};

export type V2Info = {
  session_id: string;
  valid: boolean;
  frames: number;
  width: number;
  height: number;
  fps: number | null;
  joint_names: string[];
  scope: string;
};

export type TimelineRow = Record<string, number | string | boolean | null>;

export type JointState = "shown" | "low_visibility" | "off_frame" | "missing";

export type JointRow = {
  index: number;
  joint: string;
  x: number | null;
  y: number | null;
  z: number | null;
  visibility: number | null;
  presence: number | null;
  state: JointState;
};

export type PoseQuality = {
  frames: number;
  pose_detected_frames: number;
  pose_detection_rate: number;
  missing_pose_frames: number[];
  median_visibility: number | null;
  mean_visibility: number | null;
  valid_joints_selected_frame: number;
  valid_joint_share_detected_frames: number;
  threshold: number;
  scope: string;
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

async function request<T>(method: "GET" | "POST", path: string, options: { body?: unknown; token?: string } = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  if (options.token) headers["X-Facial-Token"] = options.token;
  const response = await fetch(`/api${path}`, {
    method,
    headers,
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
  });
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

function get<T>(path: string): Promise<T> {
  return request<T>("GET", path);
}

export type JobKind = "v2_build" | "v1_extract" | "v1_analyse";

export type JobState = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted";

export type Job = {
  id: string;
  kind: JobKind;
  person: string;
  session: string;
  regenerate: boolean;
  state: JobState;
  stage: string | null;
  progress: number;
  created_at: string;
  updated_at: string;
  exit_code: number | null;
  message: string | null;
  pid: number | null;
  cancel_requested: boolean;
};

export const ACTIVE_JOB_STATES: JobState[] = ["queued", "running"];

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
  headPose: (person: string, session: string) =>
    get<HeadPose>(`/persons/${enc(person)}/sessions/${enc(session)}/v1/head-pose`),
  v2Info: (person: string, session: string) =>
    get<V2Info>(`/persons/${enc(person)}/sessions/${enc(session)}/v2`),
  v2Timeline: (person: string, session: string) =>
    get<TimelineRow[]>(`/persons/${enc(person)}/sessions/${enc(session)}/v2/timeline`),
  v2Quality: (person: string, session: string, index: number, threshold: number) =>
    get<{ frame: number; quality: PoseQuality; joints: JointRow[] }>(
      `/persons/${enc(person)}/sessions/${enc(session)}/v2/frames/${index}/quality?threshold=${threshold}`,
    ),
  jobs: () => get<Job[]>("/jobs"),
  job: (id: string) => get<Job>(`/jobs/${enc(id)}`),
  createJob: (token: string, body: { kind: JobKind; person: string; session: string; regenerate: boolean }) =>
    request<Job>("POST", "/jobs", { body, token }),
  cancelJob: (token: string, id: string) => request<Job>("POST", `/jobs/${enc(id)}/cancel`, { token }),
  v2OverlayUrl: (person: string, session: string, index: number, threshold: number, points: boolean, skeleton: boolean) =>
    `/api/persons/${enc(person)}/sessions/${enc(session)}/v2/frames/${index}/overlay.jpg?threshold=${threshold}&points=${points}&skeleton=${skeleton}`,
};
