"use client";

import { useEffect, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ACTIVE_JOB_STATES, ApiFailure, api, type Job, type JobKind, type JobState } from "../lib/api";
import { Badge, Card, Empty, ErrorState, Loading } from "../components/ui";

const TOKEN_KEY = "facial.apiToken";

const KIND_LABEL: Record<JobKind, string> = {
  v1_extract: "V1 face extraction",
  v1_analyse: "V1 analysis outputs",
  v2_build: "V2 Pose build",
};

const KIND_HELP: Record<JobKind, string> = {
  v1_extract: "Reads the recording and writes the version 1 face data and frames. Never overwrites existing data. Progress is per frame.",
  v1_analyse: "Writes the blendshape table and motion outputs from existing version 1 data. Replacing needs the box ticked.",
  v2_build: "Writes the version 2 Pose data from the recording. Needs version 1 data first. Replacing needs the box ticked.",
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

/** The token is kept for this browser tab only. It is never put in the URL. */
function useApiToken(): [string, (value: string) => void, boolean] {
  const [token, setToken] = useState("");
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    try {
      setToken(window.sessionStorage.getItem(TOKEN_KEY) ?? "");
    } catch {
      // storage unavailable (private window or blocked): the token must be typed each time
    }
    setLoaded(true);
  }, []);
  const save = (value: string) => {
    setToken(value);
    try {
      window.sessionStorage.setItem(TOKEN_KEY, value);
    } catch {
      // ignore: the token stays in memory for this page only
    }
  };
  return [token, save, loaded];
}

function errorText(error: unknown): { message: string; code?: string } {
  if (error instanceof ApiFailure) return { message: error.message, code: error.code };
  if (error instanceof Error) return { message: error.message };
  return { message: "Unknown error" };
}

export default function JobsPage() {
  const queryClient = useQueryClient();
  const [token, saveToken, tokenLoaded] = useApiToken();
  const [kind, setKind] = useState<JobKind>("v1_extract");
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
    if (kind === "v1_extract") return layout === "none";
    return layout === "v1" || layout === "v2";
  };
  const startable = (sessions.data ?? []).filter((s) => s.has_video && eligible(s.layout));
  const selected = startable.find((s) => s.session_id === session);
  const isRegeneration = (kind === "v2_build" && selected?.layout === "v2") || kind === "v1_analyse";

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

  const onStart = (event: FormEvent) => {
    event.preventDefault();
    create.mutate();
  };

  const canStart = tokenLoaded && token !== "" && person !== "" && session !== "" && !create.isPending
    && (!isRegeneration || regenerate);

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">Jobs</h1>
        <p className="text-sm text-[var(--muted)]">
          Start a version 2 Pose build for a recording. The build runs the existing command-line code in its own
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

      <Card title="Start a V2 build">
        <form onSubmit={onStart} className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <label className="flex flex-col gap-1 text-sm">
            Job
            <select
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
              value={kind}
              onChange={(e) => { setKind(e.target.value as JobKind); setSession(""); setRegenerate(false); }}
            >
              {(Object.keys(KIND_LABEL) as JobKind[]).map((k) => <option key={k} value={k}>{KIND_LABEL[k]}</option>)}
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
                  {s.session_id}{s.layout === "v2" ? " (V2 exists)" : s.layout === "v1" ? " (V1 exists)" : ""}
                </option>
              ))}
            </select>
          </label>
          <div className="flex flex-col gap-2 text-sm">
            {isRegeneration ? (
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={regenerate} onChange={(e) => setRegenerate(e.target.checked)} />
                {kind === "v2_build" ? "Replace the existing V2 data" : "Replace the existing analysis outputs"}
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
        {cancel.isError ? <div className="mb-3"><ErrorState {...errorText(cancel.error)} /></div> : null}
        {jobs.data && jobs.data.length > 0 ? <JobTable jobs={jobs.data} token={token} onCancel={(id) => cancel.mutate(id)} cancelling={cancel.isPending} /> : null}
      </Card>
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
