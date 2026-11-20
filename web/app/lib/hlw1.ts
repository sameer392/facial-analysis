/**
 * Parser for the landmark-window format HLW1, version 1 (H8.2 backend: api/video_data.py, build_window).
 *
 * Layout, little-endian:
 *   bytes 0-3   magic "HLW1"
 *   bytes 4-7   uint32 N, the length of the JSON header
 *   bytes 8..   N bytes of UTF-8 JSON header
 *   then        stream arrays, float32, C order, each at the offset its header entry gives (relative to the
 *               start of this data section). Missing values are NaN.
 *
 * Every stream array has shape [count, points, channels]:
 *   face       [count, 478, 2]  channels x, y        (normalised image coordinates)
 *   body       [count, 33, 3]   channels x, y, visibility
 *   left_hand  [count, 21, 2]   channels x, y
 *   right_hand [count, 21, 2]   channels x, y
 *
 * The parser checks the magic, version, header fields, frame count, timestamps, and every stream's dtype, shape,
 * byte count and bounds before returning. It refuses anything else, so nothing corrupt reaches the canvas.
 */

export const HLW1_MAGIC = "HLW1";
export const HLW1_VERSION = 1;
export const HLW1_MAX_COUNT = 30;

export type StreamName = "face" | "body" | "left_hand" | "right_hand";

const POINTS: Record<StreamName, number> = { face: 478, body: 33, left_hand: 21, right_hand: 21 };
const STREAM_NAMES = Object.keys(POINTS) as StreamName[];

export type Stream = {
  name: StreamName;
  channels: string[];
  /** Float32 values, row-major: frame, point, channel. NaN marks a missing value. */
  data: Float32Array;
};

export type LandmarkWindow = {
  start: number;
  count: number;
  totalFrames: number;
  frames: number[];
  timestampsMs: number[];
  faceDetected: boolean[];
  poseDetected: boolean[];
  handStatus: { left: string[] | null; right: string[] | null };
  streams: Partial<Record<StreamName, Stream>>;
  /** Why a stream is absent, as the backend declared it. */
  unavailable: Partial<Record<StreamName, string>>;
};

export class Hlw1Error extends Error {
  constructor(message: string) {
    super(message);
    this.name = "Hlw1Error";
  }
}

function fail(message: string): never {
  throw new Hlw1Error(message);
}

function isIntegerIn(value: unknown, min: number, max: number): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= min && value <= max;
}

/** Parse one window. Throws Hlw1Error for anything that does not match the version 1 layout. */
export function parseHlw1(buffer: ArrayBuffer): LandmarkWindow {
  if (buffer.byteLength < 8) fail("the landmark window is too short to be valid");
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== HLW1_MAGIC) fail("the landmark window has an unknown format");

  const view = new DataView(buffer);
  const headerLength = view.getUint32(4, true);
  const dataStart = 8 + headerLength;
  if (dataStart > buffer.byteLength) fail("the landmark window header is truncated");

  let header: Record<string, unknown>;
  try {
    header = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(8, dataStart))) as Record<string, unknown>;
  } catch {
    fail("the landmark window header is not valid JSON");
  }

  if (header.format !== "HLW") fail("the landmark window format name is not HLW");
  if (header.version !== HLW1_VERSION) fail(`unsupported landmark window version ${String(header.version)}`);

  const count = header.count;
  if (!isIntegerIn(count, 1, HLW1_MAX_COUNT)) fail("the landmark window frame count is out of range");
  const start = header.start;
  const totalFrames = header.total_frames;
  if (!isIntegerIn(start, 0, Number.MAX_SAFE_INTEGER) || !isIntegerIn(totalFrames, 1, Number.MAX_SAFE_INTEGER)) {
    fail("the landmark window position is invalid");
  }
  if (!Array.isArray(header.frames) || header.frames.length !== count) fail("the frame list does not match the count");
  const frames = header.frames as unknown[];
  frames.forEach((f, i) => {
    if (f !== (start as number) + i) fail("the frame indexes are not consecutive from the window start");
  });

  const timestampsMs = header.timestamps_ms;
  if (!Array.isArray(timestampsMs) || timestampsMs.length !== count) fail("the timestamp list does not match the count");
  for (let i = 0; i < count; i++) {
    const t = timestampsMs[i];
    if (typeof t !== "number" || !Number.isFinite(t)) fail("a timestamp is not a finite number");
    if (i > 0 && t <= (timestampsMs[i - 1] as number)) fail("the timestamps do not increase");
  }

  const streamsMeta = header.streams as Record<string, Record<string, unknown>> | undefined;
  if (!streamsMeta || typeof streamsMeta !== "object") fail("the stream list is missing");

  const dataLength = buffer.byteLength - dataStart;
  const streams: Partial<Record<StreamName, Stream>> = {};
  const unavailable: Partial<Record<StreamName, string>> = {};

  for (const name of STREAM_NAMES) {
    const meta = streamsMeta[name];
    if (!meta) fail(`the ${name} stream is missing from the header`);
    if (meta.present !== true) {
      unavailable[name] = typeof meta.reason === "string" ? meta.reason : "not available";
      continue;
    }
    const channels = meta.channels;
    if (!Array.isArray(channels) || channels.some((c) => typeof c !== "string")) fail(`the ${name} channels are invalid`);
    const channelNames = channels as string[];
    const expectedShape = [count, POINTS[name], channelNames.length];
    if (meta.dtype !== "float32") fail(`the ${name} stream is not float32`);
    if (!Array.isArray(meta.shape) || meta.shape.length !== 3 || meta.shape.some((v, i) => v !== expectedShape[i])) {
      fail(`the ${name} stream shape does not match the frame count and point count`);
    }
    const byteCount = count * POINTS[name] * channelNames.length * 4;
    if (meta.bytes !== byteCount) fail(`the ${name} stream byte count is wrong`);
    const offset = meta.offset;
    if (!isIntegerIn(offset, 0, dataLength) || (offset as number) + byteCount > dataLength) {
      fail(`the ${name} stream lies outside the data section`);
    }
    // Copy into its own buffer: a Float32Array view needs a byte offset divisible by four, which the header does not guarantee.
    const copy = buffer.slice(dataStart + (offset as number), dataStart + (offset as number) + byteCount);
    const data = new Float32Array(copy);
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (v !== v) continue;                    // NaN: a missing value, allowed
      if (!Number.isFinite(v)) fail(`the ${name} stream contains an infinite value`);
    }
    streams[name] = { name, channels: channelNames, data };
  }

  const hs = (header.hand_status ?? {}) as Record<string, unknown>;
  const handStatus = {
    left: Array.isArray(hs.left) ? (hs.left as string[]) : null,
    right: Array.isArray(hs.right) ? (hs.right as string[]) : null,
  };
  const bools = (v: unknown) => (Array.isArray(v) ? v.map((x) => x === true) : []);

  return {
    start: start as number,
    count: count as number,
    totalFrames: totalFrames as number,
    frames: frames as number[],
    timestampsMs: timestampsMs as number[],
    faceDetected: bools(header.face_detected),
    poseDetected: bools(header.pose_detected),
    handStatus,
    streams,
    unavailable,
  };
}

/** Fetch one window as raw bytes. The response must declare version 1. */
export async function fetchWindowBytes(url: string, signal: AbortSignal): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal, cache: "no-store" });
  if (!response.ok) throw new Hlw1Error(`the landmark window request failed (${response.status})`);
  const type = response.headers.get("content-type") ?? "";
  if (!type.includes(`version=${HLW1_VERSION}`)) fail("the landmark window declares an unsupported version");
  return response.arrayBuffer();
}
