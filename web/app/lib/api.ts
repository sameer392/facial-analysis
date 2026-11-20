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
  layout: "face" | "pose" | "pose_incomplete" | "none";
  layout_name: string;
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

export type FaceSummary = {
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

export type PoseInfo = {
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

export type JobKind = "face_build" | "pose_build" | "face_extract" | "face_analyse";

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

export type ModelStatus = {
  key: string;
  filename: string;
  location: string;
  expected_sha256: string;
  version: string;
  source_url: string;
  licence_status: string;
  present: boolean;
  size_bytes: number | null;
  sha256_status: "match" | "mismatch" | "missing";
  actual_sha256: string | null;
};

export type JobLogRow = {
  job_id: string;
  kind: JobKind;
  person: string;
  session: string;
  state: JobState;
  created_at: string;
  log_bytes: number | null;
};

export type JobLogTail = { job_id: string; total_lines: number; lines: string[] };

export type LayerStatus = {
  key: "audio" | "hand";
  title: string;
  implemented: boolean;
  extractor: string;
  will_show: string;
  needed: string;
};

export type Settings = {
  api_version: string;
  schema_version: string;
  data_root_name: string;
  model_folder: string;
  python_version: string;
  platform: string;
  api_token_configured: boolean;
  token_header: string;
  writes: string;
  bind_address: string;
};

const enc = encodeURIComponent;

/** Sends a request with a JSON body and the API token, and reads the same error shape as the rest of the client. */
async function send<T>(method: "POST" | "PUT", path: string, token: string, body?: unknown, raw?: Blob): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json", "X-Facial-Token": token };
  let payload: BodyInit | undefined;
  if (raw) {
    headers["Content-Type"] = "video/mp4";
    payload = raw;
  } else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const response = await fetch(`/api${path}`, { method, headers, body: payload });
  if (!response.ok) {
    let code = "http_error";
    let message = `Request failed (${response.status})`;
    try {
      const data = (await response.json()) as { code?: string; message?: string };
      code = data.code ?? code;
      message = data.message ?? message;
    } catch {
      // not JSON: keep the generic message
    }
    throw new ApiFailure(response.status, code, message);
  }
  return (await response.json()) as T;
}

export type DeleteResult = { deleted: boolean };

export type FaceTopology = {
  mesh: { edges: number[][]; colour: string };
  regions: Record<string, { edges: number[][]; colour: string }>;
};

/** Uploads a recording and reports progress (0 to 1) while the bytes are sent. */
function uploadWithProgress(path: string, token: string, file: Blob, onProgress?: (fraction: number) => void):
  Promise<{ person: string; session: string; bytes: number; uploaded: boolean }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", `/api${path}`);
    xhr.setRequestHeader("Accept", "application/json");
    xhr.setRequestHeader("Content-Type", "video/mp4");
    xhr.setRequestHeader("X-Facial-Token", token);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      let data: { code?: string; message?: string } & Record<string, unknown> = {};
      try {
        data = JSON.parse(xhr.responseText) as typeof data;
      } catch {
        // not JSON: keep the generic message
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(data as { person: string; session: string; bytes: number; uploaded: boolean });
      } else {
        reject(new ApiFailure(xhr.status, data.code ?? "http_error", data.message ?? `Request failed (${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new ApiFailure(0, "network_error", "The upload did not reach the server."));
    xhr.send(file);
  });
}

export const api = {
  health: () => get<Health>("/health"),
  persons: () => get<string[]>("/persons"),
  sessions: (person: string) => get<SessionRow[]>(`/persons/${enc(person)}/sessions`),
  faceSummary: (person: string, session: string) =>
    get<FaceSummary>(`/persons/${enc(person)}/sessions/${enc(session)}/face/summary`),
  faceSeries: (person: string, session: string, fields: string[]) =>
    get<Series>(`/persons/${enc(person)}/sessions/${enc(session)}/face/series?fields=${enc(fields.join(","))}`),
  faceBlinks: (person: string, session: string) =>
    get<Blink[]>(`/persons/${enc(person)}/sessions/${enc(session)}/face/blinks`),
  faceLandmarks: (person: string, session: string, index: number, compensated: boolean) =>
    get<LandmarkFrame>(
      `/persons/${enc(person)}/sessions/${enc(session)}/face/landmarks/${index}?compensated=${compensated}`,
    ),
  comparison: (person: string, sessions: string[]) =>
    get<Comparison>(`/persons/${enc(person)}/comparison?sessions=${enc(sessions.join(","))}`),
  faceTopology: () => get<FaceTopology>("/face/topology"),
  headPose: (person: string, session: string) =>
    get<HeadPose>(`/persons/${enc(person)}/sessions/${enc(session)}/face/head-pose`),
  poseInfo: (person: string, session: string) =>
    get<PoseInfo>(`/persons/${enc(person)}/sessions/${enc(session)}/pose`),
  poseTimeline: (person: string, session: string) =>
    get<TimelineRow[]>(`/persons/${enc(person)}/sessions/${enc(session)}/pose/timeline`),
  poseQuality: (person: string, session: string, index: number, threshold: number) =>
    get<{ frame: number; quality: PoseQuality; joints: JointRow[] }>(
      `/persons/${enc(person)}/sessions/${enc(session)}/pose/frames/${index}/quality?threshold=${threshold}`,
    ),
  models: () => get<ModelStatus[]>("/models"),
  jobLogs: (limit = 50) => get<JobLogRow[]>(`/logs/jobs?limit=${limit}`),
  jobLogTail: (id: string, lines = 200) => get<JobLogTail>(`/logs/jobs/${enc(id)}?lines=${lines}`),
  settings: () => get<Settings>("/settings"),
  layers: () => get<LayerStatus[]>("/layers"),
  jobs: () => get<Job[]>("/jobs"),
  job: (id: string) => get<Job>(`/jobs/${enc(id)}`),
  createJob: (token: string, body: { kind: JobKind; person: string; session: string; regenerate: boolean }) =>
    request<Job>("POST", "/jobs", { body, token }),
  cancelJob: (token: string, id: string) => request<Job>("POST", `/jobs/${enc(id)}/cancel`, { token }),
  poseOverlayUrl: (person: string, session: string, index: number, threshold: number, points: boolean, skeleton: boolean) =>
    `/api/persons/${enc(person)}/sessions/${enc(session)}/pose/frames/${index}/overlay.jpg?threshold=${threshold}&points=${points}&skeleton=${skeleton}`,
  createPerson: (token: string, name: string) =>
    send<{ person: string; created: boolean }>("POST", "/persons", token, { name }),
  deletePerson: (token: string, person: string, confirm: string) =>
    send<DeleteResult>("POST", `/persons/${enc(person)}/delete`, token, { confirm }),
  deleteDataset: (token: string, person: string, session: string, confirm: string) =>
    send<DeleteResult>("POST", `/persons/${enc(person)}/sessions/${enc(session)}/dataset/delete`, token, { confirm }),
  deleteVideo: (token: string, person: string, session: string, confirm: string) =>
    send<DeleteResult>("POST", `/persons/${enc(person)}/sessions/${enc(session)}/video/delete`, token, { confirm }),
  uploadVideo: (token: string, person: string, session: string, file: Blob, onProgress?: (fraction: number) => void) =>
    uploadWithProgress(`/persons/${enc(person)}/sessions/${enc(session)}/video`, token, file, onProgress),
  /** Checks a token without changing anything: a wrong token is refused (401); a right one gets past the check (400). */
  checkToken: async (token: string): Promise<"ok" | "wrong" | "unreachable"> => {
    try {
      const response = await fetch("/api/jobs/history/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Facial-Token": token },
        body: JSON.stringify({ confirm: "" }),
      });
      if (response.status === 401) return "wrong";
      return response.status === 400 ? "ok" : "unreachable";
    } catch {
      return "unreachable";
    }
  },
  renamePerson: (token: string, person: string, to: string) =>
    send<{ person: string; renamed_to: string }>("POST", `/persons/${enc(person)}/rename`, token, { to }),
  renameSession: (token: string, person: string, session: string, to: string) =>
    send<{ person: string; session: string; renamed_to: string }>(
      "POST", `/persons/${enc(person)}/sessions/${enc(session)}/rename`, token, { to }),
  clearJobHistory: (token: string, confirm: string) =>
    send<{ removed: number }>("POST", "/jobs/history/clear", token, { confirm }),
};
