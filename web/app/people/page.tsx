"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Card, Empty, ErrorState, Loading } from "../components/ui";

export default function PeoplePage() {
  const [filter, setFilter] = useState("");
  const persons = useQuery({ queryKey: ["persons"], queryFn: api.persons });
  const shown = useMemo(
    () => (persons.data ?? []).filter((p) => p.toLowerCase().includes(filter.trim().toLowerCase())),
    [persons.data, filter],
  );

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">People</h1>
        <p className="text-sm text-[var(--muted)]">People and projects with recording folders.</p>
      </header>
      <label className="flex max-w-sm flex-col gap-1 text-sm">
        Search
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter by name"
          className="rounded border border-[var(--border)] bg-transparent px-3 py-2"
        />
      </label>
      <Card>
        {persons.isPending ? <Loading label="Loading people…" /> : null}
        {persons.isError ? <ErrorState message={persons.error.message} /> : null}
        {persons.data && persons.data.length === 0 ? <Empty title="No people yet" detail="Create one from the Streamlit dashboard until uploads move to this app." /> : null}
        {persons.data && persons.data.length > 0 && shown.length === 0 ? <Empty title="No matches" /> : null}
        <ul className="divide-y divide-[var(--border)]">
          {shown.map((person) => (
            <li key={person}>
              <Link href={`/people/${encodeURIComponent(person)}`} className="block rounded px-2 py-3 text-sm hover:bg-slate-500/10 focus-visible:ring-2 focus-visible:ring-[var(--accent)] focus:outline-none">
                {person}
              </Link>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
