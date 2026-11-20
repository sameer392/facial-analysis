"use client";

import { useEffect, useRef, useState, type MutableRefObject } from "react";
import { api } from "../lib/api";
import { Card } from "./ui";

const SPEEDS = [0.25, 0.5, 1, 1.5, 2] as const;

/** A recording's current presentation time, as the browser reports it for each presented video frame. */
export type PresentedFrame = { mediaTime: number; presentedFrames: number } | null;

/** m:ss.mmm for display. */
function clock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00.000";
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${m}:${s.toFixed(3).padStart(6, "0")}`;
}

/**
 * The original recording, played by the browser's own video element (H8.3). The file is streamed by the API with
 * HTTP Range support; nothing is copied, re-encoded or downloaded whole. Frame review is a separate control.
 *
 * Before the source is set, a one-byte Range request checks that the recording exists, so a missing or unreadable
 * recording shows a clear message rather than a generic media error.
 *
 * The parent should give this component a key of person and session, so switching sessions releases the old source.
 * presentedRef is kept up to date from requestVideoFrameCallback, for the landmark overlay (H8.4). It does not
 * cause React renders.
 */
export function HandVideoPlayer({
  person,
  session,
  fpsHint,
  presentedRef,
  onElement,
}: {
  person: string;
  session: string;
  fpsHint: number | null;
  presentedRef?: MutableRefObject<PresentedFrame>;
  onElement?: (el: HTMLVideoElement | null) => void;
}) {
  const url = api.videoUrl(person, session);
  const videoRef = useRef<HTMLVideoElement | null>(null);
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

  // Check the recording exists before setting the source. The Range request asks for one byte only.
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

  // Frame clock for the overlay: records each presented frame's media time without rendering.
  useEffect(() => {
    const el = videoRef.current;
    if (!el || status !== "ready" || !presentedRef) return;
    const withFrames = el as HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number; presentedFrames: number }) => void) => number;
      cancelVideoFrameCallback?: (id: number) => void;
    };
    if (!withFrames.requestVideoFrameCallback || !withFrames.cancelVideoFrameCallback) return;
    let handle = 0;
    let active = true;
    const tick = (_now: number, meta: { mediaTime: number; presentedFrames: number }) => {
      if (!active) return;
      presentedRef.current = { mediaTime: meta.mediaTime, presentedFrames: meta.presentedFrames };
      handle = withFrames.requestVideoFrameCallback!(tick);
    };
    handle = withFrames.requestVideoFrameCallback(tick);
    return () => {
      active = false;
      withFrames.cancelVideoFrameCallback!(handle);
      presentedRef.current = null;
    };
  }, [status, presentedRef]);

  // Release the media when the player leaves the page, or the recording changes or fails. The element is captured
  // here, because React clears the ref before cleanups run.
  useEffect(() => {
    if (status !== "ready") return;
    const el = videoRef.current;
    return () => {
      if (el) {
        el.pause();
        el.removeAttribute("src");
        el.load();
      }
      onElement?.(null);
    };
    // onElement is a parent callback; the release should run only when the recording or status changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, url]);

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
          <div className="relative flex justify-center bg-black">
            <video
              ref={(el) => {
                videoRef.current = el;
                onElement?.(el);
              }}
              src={url}
              preload="metadata"
              playsInline
              className="h-auto w-full"
              style={size ? { maxWidth: size.w } : undefined}
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
          <p className="mt-2 text-xs text-[var(--muted)]">
            Playback uses the original recording, not extracted frames. The frame number is not shown during playback.
          </p>
        </>
      ) : null}
    </Card>
  );
}
