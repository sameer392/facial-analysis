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
    landmark_stability: number | null;
  }[];
  blendshape_means: Record<string, Record<string, number | null>>;
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

export type JobKind = "face_build" | "pose_build" | "hand_build" | "face_extract" | "face_analyse";

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


// ---- hand motion (H5 read-only routes). Field names follow api/hand_data.py and the H1 to H4 tables.

export type HandSummary = {
  session_id: string;
  streams: Record<string, string | undefined>;
  tracking: boolean;
  gestures: boolean;
  source: { fps: number | null; frame_count: number | null; width: number | null; height: number | null };
  models: { hand: { file: string | null; version: string | null; licence_status: string | null } };
  detected_frames: number | null;
  status_counts: { left: Record<string, number>; right: Record<string, number> };
};

export type HandPoint = [number | null, number | null, number | null];

export type HandLandmarkPage = {
  side: "left" | "right";
  total_frames: number;
  start: number;
  limit: number;
  frames: { frame: number; status: string | null; points: HandPoint[] | null }[];
};

export type HandTablePage<T> = { total: number; offset: number; limit: number; rows: T[] };

export type HandTrackRow = {
  track_id: number;
  event: string;
  frame: number;
  timestamp_ms: number | null;
  label: string;
  status: string | null;
  association: string | null;
  uncertain_reason: string | null;
  cost: number | null;
  gap_frames: number | null;
  centroid_x_px: number | null;
  centroid_y_px: number | null;
  reason: string | null;
};

/** One motion observation. Pixel fields are image pixels (normalised x * width, y * height), not physical units. */
export type HandMotionRow = {
  track_id: number;
  frame: number;
  timestamp_ms: number | null;
  label: string;
  status: string | null;
  partial_view: number | null;
  centroid_x_px: number | null;
  centroid_y_px: number | null;
  wrist_x_px: number | null;
  wrist_y_px: number | null;
  displacement_px: number | null;
  wrist_displacement_px: number | null;
  velocity_px_per_s: number | null;
  acceleration_px_per_s2: number | null;
  direction_deg: number | null;
  orientation_deg: number | null;
  orientation_change_deg: number | null;
  openness: number | null;
  motion_reason: string | null;
  [column: string]: string | number | boolean | null;
};

export type HandGestureObservationRow = {
  track_id: number;
  frame: number;
  timestamp_ms: number | null;
  label: string;
  frame_class: string;
  reason: string | null;
  quality: string;
  thumb_state: string | null;
  index_state: string | null;
  middle_state: string | null;
  ring_state: string | null;
  pinky_state: string | null;
  margin: number | null;
  event_id: string | null;
};

export type HandGestureEventRow = {
  event_id: string;
  track_id: number;
  label: string;
  gesture: string;
  start_frame: number;
  end_frame: number;
  start_timestamp_ms: number;
  end_timestamp_ms: number;
  duration_ms: number;
  supporting_observations: number;
  bridged_observations: number;
  support_quality: number;
  margin_median: number | null;
  end_reason: string;
};

export type HandTrackingQuality = {
  frames: number;
  tracks_created: number;
  tracks_terminated: number;
  interruptions: number;
  reacquisitions: number;
  observations: number;
  uncertain_observations: Record<string, number>;
  missing_frame_reasons: Record<string, number>;
  motion_reasons: Record<string, number>;
  velocity_px_per_s: { count: number; median: number | null; peak: number | null };
  timestamps: { median_interval_ms: number | null; irregular_intervals: number; frame_index_gaps: number };
  smoothing: string;
  coordinates: string;
};

export type HandGestureQuality = {
  observations: number;
  frame_classes: Record<string, number>;
  unknown_reasons: Record<string, number>;
  observation_quality: Record<string, number>;
  uncertain_tracking_observations: number;
  events: number;
  events_per_gesture: Record<string, number>;
  duration_ms: { min: number | null; median: number | null; max: number | null };
  end_reasons: Record<string, number>;
  bridged_observations: number;
  scope: string;
};

export type HandValidation = {
  session: string;
  valid: boolean;
  errors: string[];
  warnings: string[];
  metrics: Record<string, unknown>;
  provenance: {
    extraction?: Record<string, unknown>;
    hand_model?: { file?: string; sha256?: string; version?: string; licence_status?: string; licence_note?: string };
    processing_timestamp_utc?: string | null;
    manifest_sha256?: string | null;
    calibration?: string;
    source?: Record<string, unknown>;
  };
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

export type FrameBlendshapes = {
  index: number;
  time_s: number | null;
  face_detected: boolean;
  values: [string, number][];
};

export type BlendshapeSeries = {
  time_s: (number | null)[];
  values: Record<string, (number | null)[]>;
  missing: string[];
};

export type VideoFacts = {
  fps: number | null;
  frames: number;
  width: number;
  height: number;
  duration_s: number | null;
};

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
  compareAcross: (items: string[]) =>
    get<Comparison>(`/comparison?items=${enc(items.join(","))}`),
  comparison: (person: string, sessions: string[]) =>
    get<Comparison>(`/persons/${enc(person)}/comparison?sessions=${enc(sessions.join(","))}`),
  faceTopology: () => get<FaceTopology>("/face/topology"),
  faceBlendshapesFrame: (person: string, session: string, index: number) =>
    get<FrameBlendshapes>(`/persons/${enc(person)}/sessions/${enc(session)}/face/frames/${index}/blendshapes`),
  faceBlendshapesCsvUrl: (person: string, session: string) =>
    `/api/persons/${enc(person)}/sessions/${enc(session)}/face/blendshapes.csv`,
  faceBlendshapeSeries: (person: string, session: string, names: string[]) =>
    get<BlendshapeSeries>(`/persons/${enc(person)}/sessions/${enc(session)}/face/blendshapes/series?names=${enc(names.join(","))}`),
  faceVideo: (person: string, session: string) =>
    get<VideoFacts>(`/persons/${enc(person)}/sessions/${enc(session)}/face/video`),
  faceOriginalUrl: (person: string, session: string, index: number) =>
    `/api/persons/${enc(person)}/sessions/${enc(session)}/face/frames/${index}/original.jpg`,
  faceOverlayUrl: (person: string, session: string, index: number, mode: string, opacity: number, thickness: number, pointSize: number) =>
    `/api/persons/${enc(person)}/sessions/${enc(session)}/face/frames/${index}/overlay.jpg?mode=${mode}&opacity=${opacity}&thickness=${thickness}&point_radius=${pointSize}`,
  headPose: (person: string, session: string) =>
    get<HeadPose>(`/persons/${enc(person)}/sessions/${enc(session)}/face/head-pose`),
  poseRejected: () => get<{ path: string; reason: string }[]>("/pose/rejected"),
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
  handSummary: (person: string, session: string) =>
    get<HandSummary>(`/persons/${enc(person)}/sessions/${enc(session)}/hands/summary`),
  handLandmarks: (person: string, session: string, side: "left" | "right", start: number, limit: number) =>
    get<HandLandmarkPage>(`/persons/${enc(person)}/sessions/${enc(session)}/hands/landmarks?side=${side}&start=${start}&limit=${limit}`),
  handTracks: (person: string, session: string, offset: number, limit: number) =>
    get<HandTablePage<HandTrackRow>>(`/persons/${enc(person)}/sessions/${enc(session)}/hands/tracks?offset=${offset}&limit=${limit}`),
  handMotion: (person: string, session: string, offset: number, limit: number) =>
    get<HandTablePage<HandMotionRow>>(`/persons/${enc(person)}/sessions/${enc(session)}/hands/motion?offset=${offset}&limit=${limit}`),
  handGestureObservations: (person: string, session: string, offset: number, limit: number) =>
    get<HandTablePage<HandGestureObservationRow>>(`/persons/${enc(person)}/sessions/${enc(session)}/hands/gestures/observations?offset=${offset}&limit=${limit}`),
  handGestureEvents: (person: string, session: string, offset: number, limit: number) =>
    get<HandTablePage<HandGestureEventRow>>(`/persons/${enc(person)}/sessions/${enc(session)}/hands/gestures/events?offset=${offset}&limit=${limit}`),
  handQualityTracking: (person: string, session: string) =>
    get<HandTrackingQuality>(`/persons/${enc(person)}/sessions/${enc(session)}/hands/quality/tracking`),
  handQualityGestures: (person: string, session: string) =>
    get<HandGestureQuality>(`/persons/${enc(person)}/sessions/${enc(session)}/hands/quality/gestures`),
  handValidation: (person: string, session: string) =>
    get<HandValidation>(`/persons/${enc(person)}/sessions/${enc(session)}/hands/validation`),
  /** The original recording, streamed by the API with HTTP Range support (H8.2). Read-only; localhost only. */
  videoUrl: (person: string, session: string) =>
    `/api/persons/${enc(person)}/sessions/${enc(session)}/video`,
  handOverlayUrl: (person: string, session: string, index: number) =>
    `/api/persons/${enc(person)}/sessions/${enc(session)}/hands/frames/${index}/overlay.jpg`,
  clearJobHistory: (token: string, confirm: string) =>
    send<{ removed: number }>("POST", "/jobs/history/clear", token, { confirm }),
};
