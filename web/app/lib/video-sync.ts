/**
 * Synchronisation between the native video and the extracted landmarks (H8.4).
 *
 *  - frameForTime maps a presentation time to a frame index using the extracted timeline (never currentTime * fps).
 *  - WindowCache keeps a bounded set of 30-frame windows, de-duplicates requests, retries transient failures, and
 *    cancels requests that are no longer wanted.
 *  - drawFrame clears a canvas and draws the enabled landmark streams for one frame, in normalised coordinates
 *    mapped onto the canvas's CSS size.
 */

import { HAND_CONNECTIONS } from "./hand-data";
import { fetchWindowBytes, parseHlw1, type LandmarkWindow, type StreamName } from "./hlw1";

export const WINDOW_SIZE = 30;
export const MAX_CACHED_WINDOWS = 8;
const RETRIES = 2;
const RETRY_DELAY_MS = 400;
/** Tolerance when matching a presentation time to a timeline timestamp (half a millisecond). */
const TIME_TOLERANCE_MS = 0.5;

/** MediaPipe pose skeleton, copied from mediapipe.tasks.python.vision.pose_landmarker (35 edges). */
export const POSE_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10], [11, 12], [11, 13], [13, 15],
  [15, 17], [15, 19], [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20], [11, 23],
  [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32], [27, 31],
  [28, 32],
];

/** Body points below this visibility are not drawn, and no line reaches them. */
export const POSE_VISIBILITY_MIN = 0.5;

export const OVERLAY_COLOURS: Record<StreamName, string> = {
  face: "#22d3ee",       // cyan
  body: "#2dd4bf",       // teal
  left_hand: "#60a5fa",  // light blue
  right_hand: "#f472b6", // pink, to tell the hands apart
};

/** The frame that is on screen at a given presentation time: the last timeline frame at or before it. */
export function frameForTime(timestampsMs: ArrayLike<number>, mediaTimeSeconds: number): number {
  const n = timestampsMs.length;
  if (n === 0) return -1;
  const t = mediaTimeSeconds * 1000 + TIME_TOLERANCE_MS;
  let lo = 0;
  let hi = n - 1;
  if (t < timestampsMs[0]) return 0;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (timestampsMs[mid] <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** The first frame of the window that contains a frame index. */
export function windowStartFor(frame: number): number {
  return Math.floor(frame / WINDOW_SIZE) * WINDOW_SIZE;
}

type Entry =
  | { state: "loading"; controller: AbortController; attempts: number }
  | { state: "ready"; window: LandmarkWindow }
  | { state: "error"; message: string };

/**
 * A bounded cache of landmark windows keyed by window start. Requests for the same window are shared, and the
 * least recently used window is dropped when the cache is full. clear() aborts every request in flight.
 */
export class WindowCache {
  private entries = new Map<number, Entry>();
  private onChange: (() => void) | null = null;

  constructor(private readonly urlFor: (start: number) => string) {}

  setOnChange(fn: (() => void) | null) {
    this.onChange = fn;
  }

  get(start: number): Entry | undefined {
    const entry = this.entries.get(start);
    if (entry) {
      // mark as recently used
      this.entries.delete(start);
      this.entries.set(start, entry);
    }
    return entry;
  }

  /** Start loading a window unless it is already loaded, loading, or failed. */
  ensure(start: number) {
    if (this.entries.has(start)) return;
    this.load(start, 0);
  }

  private load(start: number, attempts: number) {
    const controller = new AbortController();
    this.entries.set(start, { state: "loading", controller, attempts });
    this.evict();
    fetchWindowBytes(this.urlFor(start), controller.signal)
      .then((bytes) => {
        if (this.entries.get(start)?.state !== "loading") return; // cleared or superseded meanwhile
        const window = parseHlw1(bytes);
        this.entries.set(start, { state: "ready", window });
        this.onChange?.();
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const current = this.entries.get(start);
        if (current?.state !== "loading") return;
        if (attempts < RETRIES && !(error instanceof Error && error.name === "Hlw1Error")) {
          setTimeout(() => {
            if (this.entries.get(start)?.state === "loading") this.load(start, attempts + 1);
          }, RETRY_DELAY_MS * (attempts + 1));
          return;
        }
        this.entries.set(start, {
          state: "error",
          message: error instanceof Error ? error.message : "the landmark window could not be loaded",
        });
        this.onChange?.();
      });
  }

  private evict() {
    while (this.entries.size > MAX_CACHED_WINDOWS) {
      const oldest = this.entries.keys().next().value as number;
      const entry = this.entries.get(oldest);
      if (entry?.state === "loading") entry.controller.abort();
      this.entries.delete(oldest);
    }
  }

  /** Drop everything and abort anything in flight. Used when the session changes or the page unmounts. */
  clear() {
    for (const entry of this.entries.values()) {
      if (entry.state === "loading") entry.controller.abort();
    }
    this.entries.clear();
  }

  get size(): number {
    return this.entries.size;
  }
}

export type DrawOptions = {
  enabled: Record<StreamName, boolean>;
  /** Normalised coordinates are mapped onto this CSS size. The canvas backing store is scaled separately. */
  cssWidth: number;
  cssHeight: number;
  /** The face region edges, from the API topology. Empty means points only. */
  faceEdges: [number, number][];
};

function point(data: Float32Array, frameIndex: number, points: number, channels: number, p: number) {
  const base = (frameIndex * points + p) * channels;
  return base;
}

/**
 * Draw one frame's enabled landmarks. The canvas must already be set up with the device-pixel-ratio transform.
 * Points that are missing or not finite are skipped, and no line reaches a missing point.
 */
export function drawFrame(ctx: CanvasRenderingContext2D, window: LandmarkWindow, frame: number, opts: DrawOptions): void {
  ctx.clearRect(0, 0, opts.cssWidth, opts.cssHeight);
  const i = frame - window.start;
  if (i < 0 || i >= window.count) return;

  const { cssWidth: w, cssHeight: h } = opts;
  const drawPoints = (
    stream: StreamName,
    edges: [number, number][],
    radius: number,
    visible: (data: Float32Array, base: number, channels: number) => boolean,
  ) => {
    const s = window.streams[stream];
    if (!s || !opts.enabled[stream]) return;
    const channels = s.channels.length;
    const points = s.data.length / (window.count * channels);
    const ok = (p: number) => {
      const base = point(s.data, i, points, channels, p);
      const x = s.data[base];
      const y = s.data[base + 1];
      return Number.isFinite(x) && Number.isFinite(y) && visible(s.data, base, channels);
    };
    ctx.strokeStyle = OVERLAY_COLOURS[stream];
    ctx.fillStyle = OVERLAY_COLOURS[stream];
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    for (const [a, b] of edges) {
      if (a >= points || b >= points || !ok(a) || !ok(b)) continue;
      const ba = point(s.data, i, points, channels, a);
      const bb = point(s.data, i, points, channels, b);
      ctx.moveTo(s.data[ba] * w, s.data[ba + 1] * h);
      ctx.lineTo(s.data[bb] * w, s.data[bb + 1] * h);
    }
    ctx.stroke();
    for (let p = 0; p < points; p++) {
      if (!ok(p)) continue;
      const base = point(s.data, i, points, channels, p);
      ctx.beginPath();
      ctx.arc(s.data[base] * w, s.data[base + 1] * h, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  drawPoints("face", opts.faceEdges, 1.2, () => true);
  drawPoints("body", POSE_CONNECTIONS, 2.2, (data, base) => data[base + 2] >= POSE_VISIBILITY_MIN);
  drawPoints("left_hand", HAND_CONNECTIONS, 2.4, () => true);
  drawPoints("right_hand", HAND_CONNECTIONS, 2.4, () => true);
}
