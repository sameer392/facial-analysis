"use client";

import { use, useState } from "react";
import Link from "next/link";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { api } from "../../../lib/api";
import { Card, Empty, ErrorState, Loading } from "../../../components/ui";
import { ComparisonView } from "../../../components/comparison-view";

export default function ComparePage({ params }: { params: Promise<{ person: string }> }) {
  const { person } = use(params);
  const sessions = useQuery({ queryKey: ["sessions", person], queryFn: () => api.sessions(person) });
  const faceSessions = (sessions.data ?? []).filter((s) => s.layout === "face" || s.layout === "pose").map((s) => s.session_id);
  const [picked, setPicked] = useState<string[] | null>(null);
  const chosen = picked ?? faceSessions;
  const comparison = useQuery({
    queryKey: ["comparison", person, chosen.join(",")],
    placeholderData: keepPreviousData,
    queryFn: () => api.comparison(person, chosen),
    enabled: chosen.length > 0,
  });

  const toggle = (id: string) => {
    setPicked((current) => {
      const base = current ?? faceSessions;
      return base.includes(id) ? base.filter((s) => s !== id) : [...base, id];
    });
  };

  return (
    <div className="flex flex-col gap-6">
      <header>
        <p className="text-sm text-[var(--muted)]"><Link href={`/people/${encodeURIComponent(person)}`} className="underline">{person}</Link> / compare</p>
        <h1 className="text-2xl font-semibold">Dataset comparison</h1>
        <p className="text-sm text-[var(--muted)]">
          Compare the distributions of several sessions. Shares of frames, on a shared range. To compare sessions of different people, use <Link href="/compare" className="underline">Compare people</Link>.
        </p>
      </header>

      {sessions.isPending ? <Loading label="Loading sessions…" /> : null}
      {sessions.isError ? <ErrorState message={sessions.error.message} /> : null}
      {sessions.data && faceSessions.length === 0 ? <Empty title="No analysed sessions" detail="Only sessions with face data can be compared." /> : null}

      <Card title="Sessions to compare">
        <fieldset className="flex flex-wrap gap-4 text-sm">
          <legend className="sr-only">Sessions</legend>
          {faceSessions.map((id) => (
            <label key={id} className="flex items-center gap-2">
              <input type="checkbox" checked={chosen.includes(id)} onChange={() => toggle(id)} />
              <span className="font-mono">{id}</span>
            </label>
          ))}
        </fieldset>
      </Card>

      {comparison.isPending && chosen.length > 0 ? <Loading label="Computing comparison…" /> : null}
      {comparison.isError ? <ErrorState message={comparison.error.message} code={(comparison.error as { code?: string }).code} /> : null}
      {comparison.data ? <ComparisonView data={comparison.data} /> : null}
    </div>
  );
}
