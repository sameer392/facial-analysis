"use client";

import Link from "next/link";
import { useQueries, useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Badge, Card, Empty, ErrorState, Loading } from "../components/ui";

export default function RecordingsPage() {
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

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">Recordings</h1>
        <p className="text-sm text-[var(--muted)]">Every recording across people, with its dataset status. Read-only.</p>
      </header>
      <Card>
        {loading ? <Loading label="Loading recordings…" /> : null}
        {failed ? <ErrorState message={(failed as Error).message} /> : null}
        {!loading && !failed && rows.length === 0 ? <Empty title="No recordings" detail="Add recordings to recordings/raw/<person>/." /> : null}
        {rows.length > 0 ? (
          <ul className="divide-y divide-[var(--border)] text-sm">
            {rows.map((r) => (
              <li key={`${r.person}/${r.session_id}`} className="flex items-center justify-between gap-4 py-2">
                <Link href={`/people/${encodeURIComponent(r.person)}`} className="font-mono underline">
                  {r.person} / {r.session_id}
                </Link>
                <span className="flex gap-2">
                  {r.has_video ? <Badge tone="good">video</Badge> : <Badge>no video</Badge>}
                  <Badge tone={r.layout === "v2" ? "good" : "neutral"}>{r.layout_name}</Badge>
                </span>
              </li>
            ))}
          </ul>
        ) : null}
      </Card>
    </div>
  );
}
