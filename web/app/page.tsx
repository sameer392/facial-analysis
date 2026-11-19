"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "./lib/api";
import { Badge, Card, ErrorState, Loading } from "./components/ui";

export default function OverviewPage() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const persons = useQuery({ queryKey: ["persons"], queryFn: api.persons });

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">Overview</h1>
        <p className="text-sm text-[var(--muted)]">Status of the local analysis API and the datasets it reads.</p>
      </header>

      <div className="grid gap-4 md:grid-cols-3">
        <Card title="API">
          {health.isPending ? <Loading label="Checking API…" /> : null}
          {health.isError ? <ErrorState message={health.error.message} code={(health.error as { code?: string }).code} /> : null}
          {health.data ? (
            <div className="flex flex-col gap-2 text-sm">
              <Badge tone="good">{health.data.status}</Badge>
              <p>API {health.data.api_version} · schema {health.data.schema_version}</p>
              <p className="text-[var(--muted)]">Data root: {health.data.data_root}</p>
              <p className="text-[var(--muted)]">{health.data.read_only ? "Read-only" : "Writable"}</p>
            </div>
          ) : null}
        </Card>
        <Card title="People and projects">
          {persons.isPending ? <Loading /> : null}
          {persons.isError ? <ErrorState message={persons.error.message} /> : null}
          {persons.data ? <p className="text-3xl font-semibold">{persons.data.length}</p> : null}
        </Card>
        <Card title="Phase">
          <p className="text-sm">P1: web shell and read-only pages.</p>
          <p className="mt-2 text-sm text-[var(--muted)]">Upload, jobs and deletes are not yet implemented.</p>
        </Card>
      </div>
    </div>
  );
}
