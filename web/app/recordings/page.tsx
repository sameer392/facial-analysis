"use client";

import Link from "next/link";
import { useState } from "react";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Badge, Card, Empty, ErrorState, Loading } from "../components/ui";
import { ConfirmDialog } from "../components/manage-ui";

type Target = { kind: "data" | "video"; person: string; session: string };

export default function RecordingsPage() {
  const queryClient = useQueryClient();
  const [target, setTarget] = useState<Target | null>(null);
  const persons = useQuery({ queryKey: ["persons"], queryFn: api.persons });
  const sessionQueries = useQueries({
    queries: (persons.data ?? []).map((person) => ({
      queryKey: ["sessions", person],
      queryFn: () => api.sessions(person),
    })),
  });

  const rows = (persons.data ?? []).flatMap((person, i) =>
    (sessionQueries[i]?.data ?? []).map((s) => ({ person, ...s })),
  );
  const loading = persons.isPending || sessionQueries.some((q) => q.isPending);
  const failed = persons.isError ? persons.error : sessionQueries.find((q) => q.isError)?.error;

  const remove = useMutation({
    mutationFn: ({ token, t }: { token: string; t: Target }) =>
      t.kind === "data"
        ? api.deleteDataset(token, t.person, t.session, `${t.person}/${t.session}`)
        : api.deleteVideo(token, t.person, t.session, `${t.person}/${t.session}`),
    onSuccess: () => {
      setTarget(null);
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      queryClient.invalidateQueries({ queryKey: ["persons"] });
    },
  });

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">Recordings</h1>
        <p className="text-sm text-[var(--muted)]">
          Every recording across people, with its status. Delete the processed data or the video from its row.
          Deletions are permanent.
        </p>
      </header>
      <Card>
        {loading ? <Loading label="Loading recordings…" /> : null}
        {failed ? <ErrorState message={(failed as Error).message} /> : null}
        {!loading && !failed && rows.length === 0 ? <Empty title="No recordings" detail="Upload one from a person's page." /> : null}
        {rows.length > 0 ? (
          <ul className="divide-y divide-[var(--border)] text-sm">
            {rows.map((r) => (
              <li key={`${r.person}/${r.session_id}`} className="flex flex-wrap items-center justify-between gap-3 py-2">
                <Link href={`/people/${encodeURIComponent(r.person)}`} className="font-mono underline">
                  {r.person} / {r.session_id}
                </Link>
                <span className="flex flex-wrap items-center gap-2">
                  {r.has_video ? <Badge tone="good">video</Badge> : <Badge>no video</Badge>}
                  <Badge tone={r.layout === "pose" ? "good" : "neutral"}>{r.layout_name}</Badge>
                  {r.layout !== "none" ? (
                    <button type="button" onClick={() => setTarget({ kind: "data", person: r.person, session: r.session_id })}
                      className="rounded border border-[var(--border)] px-2 py-1 text-xs">Delete data</button>
                  ) : null}
                  {r.has_video ? (
                    <button type="button" onClick={() => setTarget({ kind: "video", person: r.person, session: r.session_id })}
                      className="rounded border border-red-300 px-2 py-1 text-xs text-red-800 dark:text-red-200">Delete video</button>
                  ) : null}
                </span>
              </li>
            ))}
          </ul>
        ) : null}
      </Card>

      {target ? (
        <ConfirmDialog
          title={target.kind === "data" ? "Delete the processed data?" : "Delete the video?"}
          expected={`${target.person}/${target.session}`}
          actionLabel={target.kind === "data" ? "Delete data" : "Delete video"}
          pending={remove.isPending}
          error={remove.isError ? remove.error : undefined}
          onClose={() => { setTarget(null); remove.reset(); }}
          onConfirm={(token) => remove.mutate({ token, t: target })}
        >
          <p>
            {target.person} / {target.session}.{" "}
            {target.kind === "data"
              ? "The face and pose data is removed. The video is kept."
              : "The video is removed. The processed data is kept."}
          </p>
        </ConfirmDialog>
      ) : null}
    </div>
  );
}
