"use client";

import { useMemo, useState } from "react";
import { keepPreviousData, useQueries, useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Card, Empty, ErrorState, Loading } from "../components/ui";
import { ComparisonView } from "../components/comparison-view";

export default function ComparePeoplePage() {
  const persons = useQuery({ queryKey: ["persons"], queryFn: api.persons });
  const sessionQueries = useQueries({
    queries: (persons.data ?? []).map((p) => ({
      queryKey: ["sessions", p],
      queryFn: () => api.sessions(p),
    })),
  });

  const groups = (persons.data ?? []).map((p, i) => ({
    person: p,
    sessions: (sessionQueries[i]?.data ?? [])
      .filter((s) => s.layout === "face" || s.layout === "pose")
      .map((s) => s.session_id),
  })).filter((g) => g.sessions.length > 0);

  const [picked, setPicked] = useState<string[]>([]);
  const items = useMemo(() => picked.filter((item) => groups.some((g) => g.sessions.some((s) => `${g.person}/${s}` === item))), [picked, groups]);

  const comparison = useQuery({
    queryKey: ["comparison-across", items.join(",")],
    queryFn: () => api.compareAcross(items),
    enabled: items.length > 0,
    placeholderData: keepPreviousData,
  });

  const toggle = (item: string) => {
    setPicked((current) => (current.includes(item) ? current.filter((x) => x !== item) : [...current, item]));
  };

  const loadingSessions = sessionQueries.some((q) => q.isPending);

  return (
    <div className="flex flex-col gap-6">
      <header>
        <p className="text-sm text-[var(--muted)]">Across people</p>
        <h1 className="text-2xl font-semibold">Compare people</h1>
        <p className="text-sm text-[var(--muted)]">
          Choose sessions from any people. Shares of frames, on a shared range. Labels are person/session.
        </p>
      </header>

      {persons.isPending ? <Loading label="Loading people…" /> : null}
      {persons.isError ? <ErrorState message={persons.error.message} /> : null}
      {persons.data && groups.length === 0 && !loadingSessions ? (
        <Empty title="No analysed sessions" detail="Only sessions with face data can be compared." />
      ) : null}

      {groups.map((g) => (
        <Card key={g.person} title={g.person}>
          <fieldset className="flex flex-wrap gap-4 text-sm">
            <legend className="sr-only">Sessions for {g.person}</legend>
            {g.sessions.map((s) => {
              const item = `${g.person}/${s}`;
              return (
                <label key={item} className="flex items-center gap-2">
                  <input type="checkbox" checked={items.includes(item)} onChange={() => toggle(item)} />
                  <span className="font-mono">{s}</span>
                </label>
              );
            })}
          </fieldset>
        </Card>
      ))}

      {items.length === 0 && groups.length > 0 ? <p className="text-sm text-[var(--muted)]">Select at least one session to compare.</p> : null}
      {comparison.isPending && items.length > 0 ? <Loading label="Computing comparison…" /> : null}
      {comparison.isError ? <ErrorState message={comparison.error.message} code={(comparison.error as { code?: string }).code} /> : null}
      {comparison.data ? <ComparisonView data={comparison.data} /> : null}
    </div>
  );
}
