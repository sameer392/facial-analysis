"use client";

import { useEffect, useMemo, useRef, useState, type MutableRefObject } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { WindowCache, WINDOW_SIZE, drawFrame, frameForTime, OVERLAY_COLOURS, windowStartFor } from "../lib/video-sync";
import type { StreamName } from "../lib/hlw1";
import { Card } from "./ui";

const SPEEDS = [0.25, 0.5, 1, 1.5, 2] as const;
const STREAMS: { key: StreamName; label: string }[] = [
  { key: "face", label: "Face" },
  { key: "body", label: "Body" },
  { key: "left_hand", label: "Left hand" },
  { key: "right_hand", label: "Right hand" },
];

/** A recording's current presentation time, as the browser reports it for each presented video frame. */
export type PresentedFrame = { mediaTime: number; presentedFrames: number } | null;

type OverlayStatus = "off" | "synchronized" | "loading" | "unavailable";

/** m:ss.mmm for display. */
function clock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00.000";
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${m}:${s.toFixed(3).padStart(6, "0")}`;
}

/**
 * The original recording, played by the browser's video element, with the extracted landmarks drawn on a transparent
 * canvas over it (H8.4).
 *
 * Synchronisation: each presented video frame's media time is mapped to a frame of the extracted timeline. The
 * landmarks for that frame come from a cache of 30-frame windows. The video never waits for landmark data.
 * If the recording fails the timing check, the overlays are switched off and the video plays normally.
 *
 * The canvas is the same size as the video element's box, so landmarks cannot be stretched independently of the
 * picture. Parent components should key this component by person and session, so switching sessions starts afresh.
 */
export function HandVideoPlayer({
  person,
  session,
  fpsHint,
  presentedRef,
}: {
  person: string;
  session: string;
  fpsHint: number | null;
  presentedRef?: MutableRefObject<PresentedFrame>;
}) {
  const url = api.videoUrl(person, session);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const lastMediaTime = useRef(0);
  const lastStatus = useRef<OverlayStatus>("off");

  const [status, setStatus] = useState<"checking" | "ready" | "error">("checking");
  const [message, setMessage] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);
  const [muted, setMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  const [master, setMaster] = useState(true);
  const [enabled, setEnabled] = useState<Record<StreamName, boolean>>({ face: true, body: true, left_hand: true, right_hand: true });
  const [overlayStatus, setOverlayStatus] = useState<OverlayStatus>("off");
  const [overlayNote, setOverlayNote] = useState("");
  // Read by the render loop without re-rendering the page.
  const enabledRef = useRef({ master, enabled });
  enabledRef.current = { master, enabled };

  const timing = useQuery({
    queryKey: ["video-timing", person, session],
    queryFn: () => api.videoTiming(person, session),
    staleTime: Infinity,
    retry: false,
  });
  const timeline = useQuery({
    queryKey: ["pose-timeline", person, session],
    queryFn: () => api.poseTimeline(person, session),
    retry: false,
  });
  const topology = useQuery({ queryKey: ["face-topology"], queryFn: api.faceTopology, staleTime: Infinity, retry: false });

  const timestamps = useMemo(() => {
    if (!timeline.data) return null;
    return Float64Array.from(timeline.data.map((r) => Number(r.timestamp_ms)));
  }, [timeline.data]);
  const timestampsRef = useRef<Float64Array | null>(null);
  timestampsRef.current = timestamps;
  const faceEdgesRef = useRef<[number, number][]>([]);
  faceEdgesRef.current = topology.data ? (topology.data.regions ? Object.values(topology.data.regions).flatMap((r) => r.edges as [number, number][]) : []) : [];

  const overlayAllowed = timing.data?.status === "synchronized" && timestamps !== null;

  // Windows of landmarks, one cache per session. Cleared when the session changes or the page unmounts.
  const cache = useMemo(() => new WindowCache((start) => api.landmarkWindowUrl(person, session, start, WINDOW_SIZE)), [person, session]);
  useEffect(() => {
    const onChange = () => render(lastMediaTime.current);
    cache.setOnChange(onChange);
    return () => {
      cache.setOnChange(null);
      cache.clear();
    };
    // render is recreated each render, but reads only refs, so the effect needs to run only for a new cache
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cache]);

  // The recording must exist before the source is set. The Range request asks for one byte only.
  useEffect(() => {
    const controller = new AbortController();
    setStatus("checking");
    setMessage("");
    fetch(url, { headers: { Range: "bytes=0-0" }, signal: controller.signal })
      .then(async (response) => {
        if (response.status === 206 || response.status === 200) {
          setStatus("ready");
          return;
        }
        let text = "The recording could not be opened.";
        try {
          const body = (await response.json()) as { message?: string };
          text = body.message ?? text;
        } catch {
          // not JSON: keep the generic message
        }
        setMessage(text);
        setStatus("error");
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setMessage("The recording could not be reached. Check that the API is running.");
        setStatus("error");
      });
    return () => controller.abort();
  }, [url]);

  /** Draw the landmarks for one media time. Clears the overlay when it is off or not available. */
  function render(mediaTime: number) {
    lastMediaTime.current = mediaTime;
    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    const ts = timestampsRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !wrapper || !ctx) return;
    const cssW = wrapper.clientWidth;
    const cssH = wrapper.clientHeight;
    const { master: on, enabled: want } = enabledRef.current;
    const anyStream = STREAMS.some((s) => want[s.key]);
    if (!on || !anyStream || !overlayAllowedRef.current || !ts) {
      ctx.clearRect(0, 0, cssW, cssH);
      setOverlayState(on && anyStream ? "unavailable" : "off", "");
      return;
    }
    const frame = frameForTime(ts, mediaTime);
    const start = windowStartFor(frame);
    cache.ensure(start);
    cache.ensure(start + WINDOW_SIZE);
    if (start >= WINDOW_SIZE) cache.ensure(start - WINDOW_SIZE);
    const entry = cache.get(start);
    if (entry?.state === "ready" && frame < entry.window.start + entry.window.count) {
      drawFrame(ctx, entry.window, frame, { enabled: want, cssWidth: cssW, cssHeight: cssH, faceEdges: faceEdgesRef.current });
      setOverlayState("synchronized", "");
    } else if (entry?.state === "error") {
      ctx.clearRect(0, 0, cssW, cssH);
      setOverlayState("unavailable", entry.message);
    } else {
      ctx.clearRect(0, 0, cssW, cssH);
      setOverlayState("loading", "");
    }
  }

  // Overlay availability, read by the render loop.
  const overlayAllowedRef = useRef(false);
  overlayAllowedRef.current = overlayAllowed;

  /** Update the status text only when it changes, so the render loop does not re-render the page. */
  function setOverlayState(next: OverlayStatus, note: string) {
    if (lastStatus.current !== next) {
      lastStatus.current = next;
      setOverlayStatus(next);
    }
    setOverlayNote((prev) => (prev === note ? prev : note));
  }

  // Keep the canvas backing store matched to the video's CSS box, at the device's pixel ratio.
  useEffect(() => {
    const wrapper = wrapperRef.current;
    const canvas = canvasRef.current;
    if (!wrapper || !canvas || status !== "ready") return;
    const fit = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = wrapper.clientWidth;
      const h = wrapper.clientHeight;
      canvas.width = Math.max(1, Math.round(w * dpr));
      canvas.height = Math.max(1, Math.round(h * dpr));
      const ctx = canvas.getContext("2d");
      ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
      render(lastMediaTime.current);
    };
    const observer = new ResizeObserver(fit);
    observer.observe(wrapper);
    fit();
    return () => observer.disconnect();
    // render reads refs only, so it does not need to be a dependency
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, size]);

  // Drive the overlay from the video's presented frames. Where requestVideoFrameCallback is missing, follow the
  // video's own currentTime each animation frame. Neither path uses a timer of its own.
  useEffect(() => {
    const el = videoRef.current;
    if (!el || status !== "ready") return;
    const withFrames = el as HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number; presentedFrames: number }) => void) => number;
      cancelVideoFrameCallback?: (id: number) => void;
    };
    let active = true;
    let handle = 0;
    const onSeeked = () => render(el.currentTime);
    el.addEventListener("seeked", onSeeked);

    if (withFrames.requestVideoFrameCallback && withFrames.cancelVideoFrameCallback) {
      const tick = (_now: number, meta: { mediaTime: number; presentedFrames: number }) => {
        if (!active) return;
        if (presentedRef) presentedRef.current = { mediaTime: meta.mediaTime, presentedFrames: meta.presentedFrames };
        render(meta.mediaTime);
        handle = withFrames.requestVideoFrameCallback!(tick);
      };
      handle = withFrames.requestVideoFrameCallback(tick);
      return () => {
        active = false;
        el.removeEventListener("seeked", onSeeked);
        withFrames.cancelVideoFrameCallback!(handle);
        if (presentedRef) presentedRef.current = null;
      };
    }

    const frameLoop = () => {
      if (!active) return;
      render(el.currentTime);
      handle = requestAnimationFrame(frameLoop);
    };
    handle = requestAnimationFrame(frameLoop);
    return () => {
      active = false;
      el.removeEventListener("seeked", onSeeked);
      cancelAnimationFrame(handle);
    };
    // render reads refs only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  // Release the media when the player leaves the page, or the recording changes or fails.
  useEffect(() => {
    if (status !== "ready") return;
    const el = videoRef.current;
    return () => {
      if (el) {
        el.pause();
        el.removeAttribute("src");
        el.load();
      }
    };
  }, [status, url]);

  // Redraw when the overlay settings change, even while paused.
  useEffect(() => {
    render(lastMediaTime.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [master, enabled, overlayAllowed]);

  const toggle = () => {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused) void el.play();
    else el.pause();
  };
  const seek = (value: number) => {
    const el = videoRef.current;
    if (el) el.currentTime = value;
    setCurrent(value);
  };
  const step = (direction: 1 | -1) => {
    const el = videoRef.current;
    if (!el) return;
    el.pause();
    const frameSeconds = fpsHint && fpsHint > 0 ? 1 / fpsHint : 1 / 30;
    el.currentTime = Math.min(Math.max(el.currentTime + direction * frameSeconds, 0), el.duration || 0);
  };
  const changeSpeed = (value: (typeof SPEEDS)[number]) => {
    setSpeed(value);
    if (videoRef.current) videoRef.current.playbackRate = value;
  };
  const changeVolume = (value: number) => {
    setVolume(value);
    if (videoRef.current) {
      videoRef.current.volume = value;
      videoRef.current.muted = value === 0;
    }
    setMuted(value === 0);
  };
  const toggleMute = () => {
    const el = videoRef.current;
    const next = !muted;
    setMuted(next);
    if (el) el.muted = next;
  };
  const setStream = (key: StreamName, on: boolean) => setEnabled((prev) => ({ ...prev, [key]: on }));

  const statusText: Record<OverlayStatus, string> = {
    off: "Overlay off",
    synchronized: "Synchronised with the video",
    loading: "Loading landmarks…",
    unavailable: "Overlay unavailable for these frames",
  };

  return (
    <Card title="Recording playback">
      {status === "checking" ? <p className="text-sm text-[var(--muted)]">Checking the recording…</p> : null}
      {status === "error" ? (
        <div role="alert" className="rounded border border-red-300 bg-red-50 p-3 text-sm text-red-900">
          <p className="font-medium">The recording cannot be played.</p>
          <p>{message}</p>
        </div>
      ) : null}

      {status === "ready" ? (
        <>
          <div ref={wrapperRef} className="relative mx-auto bg-black" style={size ? { maxWidth: size.w } : undefined}>
            <video
              ref={(el) => {
                videoRef.current = el;
              }}
              src={url}
              preload="metadata"
              playsInline
              className="block h-auto w-full"
              onLoadedMetadata={(e) => {
                const el = e.currentTarget;
                setSize({ w: el.videoWidth, h: el.videoHeight });
                setDuration(el.duration);
                el.playbackRate = speed;
                el.volume = volume;
                el.muted = muted;
              }}
              onDurationChange={(e) => setDuration(e.currentTarget.duration)}
              onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => setPlaying(false)}
              onWaiting={() => setLoading(true)}
              onPlaying={() => setLoading(false)}
              onCanPlay={() => setLoading(false)}
              onError={(e) => {
                const code = e.currentTarget.error?.code;
                setStatus("error");
                setMessage(code === 4
                  ? "This file cannot be decoded by the browser. The recording may be unsupported."
                  : "Playback failed. Try reloading the page.");
              }}
            />
            <canvas
              ref={canvasRef}
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 h-full w-full"
            />
            {loading ? (
              <span className="absolute right-3 top-3 rounded bg-black/70 px-2 py-1 text-xs text-white">Buffering…</span>
            ) : null}
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
            <button type="button" onClick={toggle}
              className="min-w-20 rounded border border-[var(--border)] px-3 py-1" aria-pressed={playing}>
              {playing ? "Pause" : "Play"}
            </button>
            <button type="button" onClick={() => step(-1)} disabled={playing || current <= 0}
              className="rounded border border-[var(--border)] px-2 py-1 disabled:opacity-50">
              Back one frame
            </button>
            <button type="button" onClick={() => step(1)} disabled={playing || current >= duration}
              className="rounded border border-[var(--border)] px-2 py-1 disabled:opacity-50">
              Forward one frame
            </button>
            <label className="flex items-center gap-2">
              Speed
              <select value={speed} onChange={(e) => changeSpeed(Number(e.target.value) as (typeof SPEEDS)[number])}
                className="rounded border border-[var(--border)] bg-transparent px-2 py-1">
                {SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}
              </select>
            </label>
            <button type="button" onClick={toggleMute} className="rounded border border-[var(--border)] px-2 py-1"
              aria-pressed={muted}>
              {muted ? "Unmute" : "Mute"}
            </button>
            <label className="flex items-center gap-2">
              Volume
              <input type="range" min={0} max={1} step={0.05} value={muted ? 0 : volume}
                onChange={(e) => changeVolume(Number(e.target.value))} aria-label="Volume" />
            </label>
          </div>

          <div className="mt-3">
            <input
              type="range"
              min={0}
              max={duration || 0}
              step={0.001}
              value={Math.min(current, duration || 0)}
              onChange={(e) => seek(Number(e.target.value))}
              aria-label="Playback position"
              className="w-full"
            />
            <p className="mt-1 text-sm">
              {clock(current)} / {clock(duration)}
              {size ? <span className="text-[var(--muted)]"> · {size.w} × {size.h} px</span> : null}
            </p>
          </div>

          <div className="mt-4 rounded border border-[var(--border)] p-3 text-sm">
            <div className="flex flex-wrap items-center gap-4">
              <label className="flex items-center gap-2 font-medium">
                <input type="checkbox" checked={master} disabled={!overlayAllowed}
                  onChange={(e) => setMaster(e.target.checked)} />
                Landmark overlay
              </label>
              {STREAMS.map((s) => (
                <label key={s.key} className="flex items-center gap-2">
                  <span aria-hidden="true" className="inline-block h-3 w-3 rounded-full"
                    style={{ background: OVERLAY_COLOURS[s.key] }} />
                  <input type="checkbox" checked={enabled[s.key]} disabled={!overlayAllowed || !master}
                    onChange={(e) => setStream(s.key, e.target.checked)} />
                  {s.label}
                </label>
              ))}
            </div>

            <p className="mt-2" aria-live="polite">
              <span className="font-medium">
                {!overlayAllowed && timing.isPending ? "Checking timing…" : null}
                {!overlayAllowed && !timing.isPending ? "Timing verification failed" : null}
                {overlayAllowed ? statusText[overlayStatus] : null}
              </span>
              {overlayNote ? <span className="text-[var(--muted)]"> · {overlayNote}</span> : null}
            </p>
            {timing.data && timing.data.status !== "synchronized" ? (
              <p role="alert" className="mt-1 text-sm text-red-800">
                Landmark overlays are disabled for this recording: {timing.data.reason || "the frame times could not be verified"}.
                Playback is unaffected.
              </p>
            ) : null}
            {timing.isError || timeline.isError ? (
              <p className="mt-1 text-sm text-[var(--muted)]">
                Landmark overlays need the pose data for this session. Playback is unaffected.
              </p>
            ) : null}
          </div>

          <p className="mt-2 text-xs text-[var(--muted)]">
            Playback uses the original recording. Landmarks come from the extracted data, matched to the frame on screen.
          </p>
        </>
      ) : null}
    </Card>
  );
}
