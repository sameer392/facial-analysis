"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiFailure } from "../lib/api";
import { Card, Empty, ErrorState, Loading } from "../components/ui";
import { HandSessionSelector } from "../components/hand-session-selector";
import { HandJobPanel } from "../components/hand-job-panel";
import { HandLandmarkViewer } from "../components/hand-landmark-viewer";
import { HandMotionCharts } from "../components/hand-motion-charts";
import { HandGesturePanel } from "../components/hand-gesture-panel";
import { HandQualityPanel } from "../components/hand-quality-panel";

const SECTIONS: [string, string][] = [
  ["processing", "Processing"],
  ["frames", "Frames and landmarks"],
  ["motion", "Motion"],
  ["gestures", "Gestures"],
  ["quality", "Dataset quality"],
];

export default function HandsPage() {
  const [person, setPerson] = useState("");
  const [session, setSession] = useState("");

  const sessions = useQuery({
    queryKey: ["sessions", person],
    queryFn: () => api.sessions(person),
    enabled: person !== "",
  });
  const pose = (sessions.data ?? []).find((s) => s.session_id === session)?.layout === "pose";

  const summary = useQuery({
    queryKey: ["hand-summary", person, session],
    queryFn: () => api.handSummary(person, session),
    enabled: person !== "" && session !== "",
    retry: false,
  });
  // "No hand analysis yet" is a normal state, not a failure.
  const noAnalysis = summary.isError && summary.error instanceof ApiFailure && summary.error.code === "no_hand_analysis";
  const hasAnalysis = summary.data !== undefined;

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">Hand motion analysis</h1>
        <p className="text-sm text-[var(--muted)]">
          Hand landmarks, tracking, motion, gesture events and dataset quality for one recording. Pixel measures are in
          image pixels. Gesture rules are not calibrated on labelled footage, so their output is not an accuracy figure.
        </p>
        <nav aria-label="On this page" className="mt-3 flex flex-wrap gap-3 text-sm">
          {SECTIONS.map(([id, label]) => (
            <a key={id} href={`#${id}`} className="underline">{label}</a>
          ))}
        </nav>
      </header>

      <section id="selection" className="scroll-mt-6">
        <Card title="Recording">
          <HandSessionSelector person={person} session={session} onPerson={setPerson} onSession={setSession} />
          {session !== "" && summary.data ? (
            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
              <div><dt className="text-[var(--muted)]">Video</dt><dd>{summary.data.source.width ?? "?"} × {summary.data.source.height ?? "?"} px</dd></div>
              <div><dt className="text-[var(--muted)]">Frame rate</dt><dd>{summary.data.source.fps ?? "?"} fps</dd></div>
              <div><dt className="text-[var(--muted)]">Frames</dt><dd>{summary.data.source.frame_count ?? "?"}</dd></div>
              <div><dt className="text-[var(--muted)]">Hand analysis</dt><dd>{summary.data.streams.hands ?? "not run"}</dd></div>
            </dl>
          ) : null}
        </Card>
        {person !== "" && session === "" ? <div className="mt-4"><Empty title="Choose a recording" detail="Pick a person and a session to see its hand analysis." /></div> : null}
        {session !== "" && summary.isPending ? <div className="mt-4"><Loading label="Loading the hand analysis status…" /></div> : null}
        {session !== "" && summary.isError && !noAnalysis ? (
          <div className="mt-4"><ErrorState message={summary.error.message} code={(summary.error as { code?: string }).code} /></div>
        ) : null}
        {session !== "" && noAnalysis ? (
          <div className="mt-4"><Empty title="No hand analysis yet" detail={pose ? "Start it below." : "This session needs pose data first."} /></div>
        ) : null}
      </section>

      {session !== "" ? (
        <section id="processing" className="scroll-mt-6">
          <HandJobPanel person={person} session={session} pose={pose} summary={summary.data} />
        </section>
      ) : null}

      {session !== "" && hasAnalysis && summary.data ? (
        <section id="frames" className="scroll-mt-6">
          <HandLandmarkViewer
            person={person}
            session={session}
            totalFrames={summary.data.source.frame_count ?? 0}
            fps={summary.data.source.fps}
          />
        </section>
      ) : null}

      {session !== "" && hasAnalysis && summary.data?.tracking ? (
        <section id="motion" className="scroll-mt-6">
          <HandMotionCharts person={person} session={session} />
        </section>
      ) : null}
      {session !== "" && hasAnalysis && summary.data && !summary.data.tracking ? (
        <section id="motion" className="scroll-mt-6">
          <Card title="Motion"><p className="text-sm">Hand tracking has not run for this session yet. It runs as part of Start hand analysis.</p></Card>
        </section>
      ) : null}

      {session !== "" && hasAnalysis && summary.data?.gestures ? (
        <section id="gestures" className="scroll-mt-6">
          <HandGesturePanel person={person} session={session} />
        </section>
      ) : null}
      {session !== "" && hasAnalysis && summary.data && !summary.data.gestures ? (
        <section id="gestures" className="scroll-mt-6">
          <Card title="Gestures"><p className="text-sm">Gesture classification has not run for this session yet.</p></Card>
        </section>
      ) : null}

      {session !== "" && hasAnalysis && summary.data ? (
        <section id="quality" className="scroll-mt-6">
          <HandQualityPanel person={person} session={session} summary={summary.data} />
        </section>
      ) : null}
    </div>
  );
}
