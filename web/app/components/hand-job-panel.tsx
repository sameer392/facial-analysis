"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ACTIVE_JOB_STATES, api, type HandSummary, type Job } from "../lib/api";
import { useApiToken } from "../lib/token";
import { HAND_STAGES, stageStates, type StageState } from "../lib/hand-data";
import { errorText, NeedsToken } from "./manage-ui";
import { Badge, Card, ErrorState } from "./ui";

const STAGE_BADGE: Record<StageState, "neutral" | "good" | "warn"> = {
  done: "good",
  running: "neutral",
  pending: "neutral",
  "not run": "neutral",
  failed: "warn",
  cancelled: "warn",
};

/**
 * Start, follow and cancel the hand_build job for one session. Shows each pipeline stage separately, so a
 * failed later stage is never shown as a success. Existing hand outputs are not regenerated unless ticked.
 */
export function HandJobPanel({
  person,
  session,
  pose,
  summary,
}: {
  person: string;
  session: string;
  pose: boolean;
  summary: HandSummary | undefined;
}) {
  const queryClient = useQueryClient();
  const [token, , tokenLoaded] = useApiToken();

  // Every cached hand query for this session. Called when a job ends, so no page keeps data from before a run.
  // That includes the validation result, which is otherwise cached indefinitely.
  const refreshHandData = () => {
    queryClient.invalidateQueries({
      predicate: (q) => typeof q.queryKey[0] === "string" && String(q.queryKey[0]).startsWith("hand") &&
        q.queryKey[1] === person && q.queryKey[2] === session,
    });
  };
  const [regenerate, setRegenerate] = useState(false);
  const hasOutputs = Boolean(summary && (summary.streams.hands ?? "") !== "");

  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: api.jobs,
    refetchInterval: (query) => (query.state.data?.some((j) => ACTIVE_JOB_STATES.includes(j.state)) ? 2000 : false),
  });
  const current: Job | null = (jobs.data ?? [])
    .filter((j) => j.kind === "hand_build" && j.person === person && j.session === session)
    .sort((a, b) => (a.created_at < b.created_at ? 1 : -1))[0] ?? null;
  const active = current !== null && ACTIVE_JOB_STATES.includes(current.state);

  // Live progress from the job's event stream, with polling as the fallback.
  const activeId = active && current ? current.id : "";
  useEffect(() => {
    if (activeId === "") return;
    const source = new EventSource(`/api/jobs/${activeId}/events`);
    source.onmessage = (event) => {
      const updated = JSON.parse(event.data) as Job;
      queryClient.setQueryData<Job[]>(["jobs"], (old) => old?.map((j) => (j.id === updated.id ? updated : j)));
      if (!ACTIVE_JOB_STATES.includes(updated.state)) {
        source.close();
        queryClient.invalidateQueries({ queryKey: ["jobs"] });
        refreshHandData();
      }
    };
    source.onerror = () => source.close();
    return () => source.close();
    // refreshHandData depends only on the query client and the session identity
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId, queryClient, person, session]);

  // A job that finished while this page was open (polling path) also refreshes the cached hand data.
  const lastState = current?.state ?? "";
  useEffect(() => {
    if (lastState === "succeeded" || lastState === "failed" || lastState === "cancelled" || lastState === "interrupted") {
      refreshHandData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastState, current?.id]);

  const create = useMutation({
    mutationFn: () => api.createJob(token, { kind: "hand_build", person, session, regenerate }),
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

  const blockedByOutputs = hasOutputs && !regenerate;
  const canStart = tokenLoaded && token !== "" && pose && !active && !blockedByOutputs && !create.isPending;
  const states = stageStates(current);
  const error = create.isError ? errorText(create.error) : cancel.isError ? errorText(cancel.error) : null;

  return (
    <Card title="Processing">
      <p className="mb-3 text-sm text-[var(--muted)]">
        Runs the hand pipeline on this session&apos;s recording: landmarks, tracking, gestures and validation. It needs
        pose data first. It never rebuilds face or pose data.
      </p>
      <NeedsToken token={token} loaded={tokenLoaded} />
      {!pose ? (
        <p className="mb-3 text-sm">This session has no pose data yet. Run Face data and Pose data on the Jobs page first.</p>
      ) : null}

      {hasOutputs ? (
        <label className="mb-3 flex items-center gap-2 text-sm">
          <input type="checkbox" checked={regenerate} onChange={(e) => setRegenerate(e.target.checked)} />
          Replace the existing hand outputs (they are kept for rollback until the new run publishes)
        </label>
      ) : null}

      <div className="flex flex-wrap gap-3">
        <button
          type="button"
          disabled={!canStart}
          onClick={() => create.mutate()}
          className="rounded border border-[var(--border)] px-3 py-1 text-sm disabled:opacity-50"
        >
          Start hand analysis
        </button>
        {active && current ? (
          <button
            type="button"
            disabled={cancel.isPending || current.cancel_requested}
            onClick={() => cancel.mutate(current.id)}
            className="rounded border border-[var(--border)] px-3 py-1 text-sm disabled:opacity-50"
          >
            {current.cancel_requested ? "Cancelling…" : "Cancel"}
          </button>
        ) : null}
      </div>
      {blockedByOutputs ? (
        <p className="mt-2 text-sm text-[var(--muted)]">Hand outputs already exist. Tick the box above to replace them.</p>
      ) : null}
      {error ? <div className="mt-3"><ErrorState message={error.message} code={error.code} /></div> : null}

      {current ? (
        <div className="mt-4 text-sm">
          <dl className="grid grid-cols-2 gap-2 md:grid-cols-4">
            <div><dt className="text-[var(--muted)]">Job</dt><dd className="font-mono text-xs">{current.id.slice(0, 8)}</dd></div>
            <div><dt className="text-[var(--muted)]">Status</dt><dd>{current.state}</dd></div>
            <div><dt className="text-[var(--muted)]">Progress</dt><dd>{Math.round(current.progress * 100)} %</dd></div>
            <div><dt className="text-[var(--muted)]">Current stage</dt><dd>{current.stage ?? "waiting to start"}</dd></div>
          </dl>
          <div
            className="mt-3 h-2 w-full overflow-hidden rounded bg-[var(--border)]"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(current.progress * 100)}
            aria-label="Hand analysis progress"
          >
            <div className="h-full bg-[var(--muted)]" style={{ width: `${Math.round(current.progress * 100)}%` }} />
          </div>
          <ol className="mt-3 flex flex-col gap-1">
            {HAND_STAGES.map((stage, i) => (
              <li key={stage.key} className="flex items-center justify-between gap-2 border-t border-[var(--border)] py-1">
                <span>{stage.label}</span>
                <Badge tone={STAGE_BADGE[states[i]]}>{states[i]}</Badge>
              </li>
            ))}
          </ol>
          {current.state === "failed" || current.state === "interrupted" || current.state === "cancelled" ? (
            <p className="mt-2 text-sm">
              {current.state === "cancelled"
                ? "Cancelled. Earlier stages that finished are kept; the stage that was running was not published."
                : `Stopped: ${current.message ?? "see the job log"}`}
            </p>
          ) : null}
          {current.state === "succeeded" ? <p className="mt-2 text-sm">All four stages finished.</p> : null}
        </div>
      ) : (
        <p className="mt-4 text-sm text-[var(--muted)]">No hand analysis job has run for this session in this API session.</p>
      )}
    </Card>
  );
}
