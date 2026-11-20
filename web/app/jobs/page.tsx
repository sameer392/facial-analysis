"use client";

import { useEffect, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ACTIVE_JOB_STATES, ApiFailure, api, type Job, type JobKind, type JobState } from "../lib/api";
import { useApiToken } from "../lib/token";
import { Badge, Card, Empty, ErrorState, Loading } from "../components/ui";
import { ConfirmDialog } from "../components/manage-ui";


// Jobs offered on this page. Older single-step jobs still show their names in the job list.
const OFFERED: JobKind[] = ["face_build", "pose_build"];

const KIND_LABEL: Record<JobKind, string> = {
  face_build: "Face data",
  pose_build: "Pose data",
  face_extract: "Face extraction (older job)",
  face_analyse: "Face analysis outputs (older job)",
};

const KIND_HELP: Record<JobKind, string> = {
  face_build: "Reads the recording and writes the face data: landmarks, blendshapes and motion measures. Never overwrites existing face data. Progress is per frame, then per stage.",
  pose_build: "Writes the pose data from the recording. Needs face data first. Replacing needs the box ticked.",
  face_extract: "Reads the recording and writes the face data and frames. Never overwrites existing data. Progress is per frame.",
  face_analyse: "Writes the blendshape table and motion outputs from existing face data. Replacing needs the box ticked.",
};

const STATE_LABEL: Record<JobState, string> = {
  queued: "Queued",
  running: "Running",
  succeeded: "Succeeded",
  failed: "Failed",
  cancelled: "Cancelled",
  interrupted: "Interrupted",
};

function stateTone(state: JobState): "neutral" | "good" | "warn" {
  if (state === "succeeded") return "good";
  if (state === "failed" || state === "interrupted") return "warn";
  return "neutral";
}

function errorText(error: unknown): { message: string; code?: string } {
  if (error instanceof ApiFailure) return { message: error.message, code: error.code };
  if (error instanceof Error) return { message: error.message };
  return { message: "Unknown error" };
}

export default function JobsPage() {
  const queryClient = useQueryClient();
  const [token, saveToken, tokenLoaded] = useApiToken();
  const [kind, setKind] = useState<JobKind>("face_build");
  const [person, setPerson] = useState("");
  const [session, setSession] = useState("");
  const [regenerate, setRegenerate] = useState(false);

  const persons = useQuery({ queryKey: ["persons"], queryFn: api.persons });
  const sessions = useQuery({
    queryKey: ["sessions", person],
    queryFn: () => api.sessions(person),
    enabled: person !== "",
  });
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: api.jobs,
    refetchInterval: (query) => (query.state.data?.some((j) => ACTIVE_JOB_STATES.includes(j.state)) ? 2000 : false),
  });

  // Which sessions each kind can start on, following the script rules. The server checks again.
  const eligible = (layout: string) => {
    if (kind === "face_build" || kind === "face_extract") return layout === "none";
    return layout === "face" || layout === "pose";
  };
  const startable = (sessions.data ?? []).filter((s) => s.has_video && eligible(s.layout));
  const selected = startable.find((s) => s.session_id === session);
  const isRegeneration = (kind === "pose_build" && selected?.layout === "pose") || kind === "face_analyse";

  const create = useMutation({
    mutationFn: () => api.createJob(token, { kind, person, session, regenerate }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const cancel = useMutation({
    mutationFn: (id: string) => api.cancelJob(token, id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const [clearing, setClearing] = useState(false);
  const clearHistory = useMutation({
    mutationFn: (t: string) => api.clearJobHistory(t, "clear job history"),
    onSuccess: () => {
      setClearing(false);
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const onStart = (event: FormEvent) => {
    event.preventDefault();
    create.mutate();
  };

  // The first missing requirement, shown next to the button so a disabled button explains itself.
  const blocker =
    !tokenLoaded ? null
    : token === "" ? "Enter the API token above to start a build."
    : person === "" ? "Choose a person."
    : session === "" ? "Choose a session."
    : isRegeneration && !regenerate ? "This session already has data for this job. Tick “Replace” to start it again."
    : null;
  const canStart = blocker === null && !create.isPending;

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">Jobs</h1>
        <p className="text-sm text-[var(--muted)]">
          Start a pose build for a recording. The build runs the existing command-line code in its own
          process. Progress is shown per stage, not per frame. Starting and cancelling need the API token.
        </p>
      </header>

      <Card title="API token">
        <div className="flex flex-col gap-2 text-sm md:flex-row md:items-end">
          <label className="flex flex-1 flex-col gap-1">
            Token from the API console (kept for this tab only)
            <input
              type="password"
              autoComplete="off"
              value={token}
              onChange={(e) => saveToken(e.target.value)}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono"
              aria-describedby="token-help"
            />
          </label>
          <p id="token-help" className="text-xs text-[var(--muted)]">
            {token ? "Token set for this tab." : "No token set. Starting and cancelling are disabled."}
          </p>
        </div>
      </Card>

      <Card title="Start a job">
        <form onSubmit={onStart} className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <label className="flex flex-col gap-1 text-sm">
            Job
            <select
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
              value={kind}
              onChange={(e) => { setKind(e.target.value as JobKind); setSession(""); setRegenerate(false); }}
            >
              {OFFERED.map((k) => <option key={k} value={k}>{KIND_LABEL[k]}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-sm">
            Person
            <select
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
              value={person}
              onChange={(e) => { setPerson(e.target.value); setSession(""); setRegenerate(false); }}
            >
              <option value="">Choose a person</option>
              {(persons.data ?? []).map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-sm">
            Session
            <select
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
              value={session}
              onChange={(e) => { setSession(e.target.value); setRegenerate(false); }}
              disabled={person === ""}
            >
              <option value="">Choose a session</option>
              {startable.map((s) => (
                <option key={s.session_id} value={s.session_id}>
                  {s.session_id}{s.layout === "pose" ? " (pose data exists)" : s.layout === "face" ? " (face data exists)" : ""}
                </option>
              ))}
            </select>
          </label>
          <div className="flex flex-col gap-2 text-sm">
            {isRegeneration ? (
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={regenerate} onChange={(e) => setRegenerate(e.target.checked)} />
                {kind === "pose_build" ? "Replace the existing pose data" : "Replace the existing analysis outputs"}
              </label>
            ) : null}
            <button
              type="submit"
              disabled={!canStart}
              className="rounded border border-[var(--border)] px-3 py-1.5 font-medium disabled:opacity-50"
            >
              {create.isPending ? "Starting…" : "Start build"}
            </button>
          </div>
        </form>
        <p className="mt-3 text-xs text-[var(--muted)]">{KIND_HELP[kind]}</p>
        {blocker ? <p className="mt-2 text-sm font-medium" role="status">{blocker}</p> : null}
        {create.isError ? (
          <div className="mt-4">
            <ErrorState {...errorText(create.error)} />
          </div>
        ) : null}
        {create.isSuccess ? (
          <p className="mt-3 text-sm">{KIND_LABEL[create.data.kind]} queued for {create.data.person} / {create.data.session}.</p>
        ) : null}
      </Card>

      <Card title="Recent jobs">
        {jobs.isPending ? <Loading label="Loading jobs…" /> : null}
        {jobs.isError ? <ErrorState {...errorText(jobs.error)} /> : null}
        {jobs.data && jobs.data.length === 0 ? (
          <Empty title="No jobs yet" detail="Jobs you start appear here." />
        ) : null}
        <div className="mb-3 flex justify-end">
          <button type="button" onClick={() => setClearing(true)} disabled={!jobs.data?.some((j) => !ACTIVE_JOB_STATES.includes(j.state))}
            className="rounded border border-[var(--border)] px-2 py-1 text-xs disabled:opacity-40">Clear finished jobs</button>
        </div>
        {cancel.isError ? <div className="mb-3"><ErrorState {...errorText(cancel.error)} /></div> : null}
        {jobs.data && jobs.data.length > 0 ? <JobTable jobs={jobs.data} token={token} onCancel={(id) => cancel.mutate(id)} cancelling={cancel.isPending} /> : null}
      </Card>
      {clearing ? (
        <ConfirmDialog
          title="Clear finished jobs?"
          expected="clear job history"
          actionLabel="Clear finished jobs"
          pending={clearHistory.isPending}
          error={clearHistory.isError ? clearHistory.error : undefined}
          onClose={() => { setClearing(false); clearHistory.reset(); }}
          onConfirm={(t) => clearHistory.mutate(t)}
        >
          <p>Removes finished job records and their logs. Jobs that are still running are kept.</p>
        </ConfirmDialog>
      ) : null}
    </div>
  );
}

function JobTable({ jobs, token, onCancel, cancelling }: { jobs: Job[]; token: string; onCancel: (id: string) => void; cancelling: boolean }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-[var(--muted)]">
          <tr>
            <th scope="col" className="py-2 font-medium">Started</th>
            <th scope="col" className="py-2 font-medium">Recording</th>
            <th scope="col" className="py-2 font-medium">State</th>
            <th scope="col" className="py-2 font-medium">Stage</th>
            <th scope="col" className="py-2 font-medium">Message</th>
            <th scope="col" className="py-2 font-medium"><span className="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => {
            const active = ACTIVE_JOB_STATES.includes(job.state);
            return (
              <tr key={job.id} className="border-t border-[var(--border)] align-top">
                <td className="py-2 whitespace-nowrap">{new Date(job.created_at).toLocaleString()}</td>
                <td className="py-2 font-mono">{job.person} / {job.session}<br /><span className="font-sans text-xs text-[var(--muted)]">{KIND_LABEL[job.kind]}{job.regenerate ? ", replace" : ""}</span></td>
                <td className="py-2"><Badge tone={stateTone(job.state)}>{STATE_LABEL[job.state]}</Badge></td>
                <td className="py-2">
                  {job.stage ? `${job.stage} (${Math.round(job.progress * 100)} %)` : "n/a"}
                </td>
                <td className="py-2 max-w-md break-words">{job.message ?? ""}</td>
                <td className="py-2">
                  {active ? (
                    <button
                      type="button"
                      onClick={() => onCancel(job.id)}
                      disabled={!token || cancelling || job.cancel_requested}
                      className="rounded border border-[var(--border)] px-2 py-1 disabled:opacity-50"
                    >
                      {job.cancel_requested ? "Cancelling…" : "Cancel"}
                    </button>
                  ) : null}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
