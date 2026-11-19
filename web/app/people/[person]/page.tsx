"use client";

import { use } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../lib/api";
import { Badge, Card, Empty, ErrorState, Loading } from "../../components/ui";

const LAYOUT_LABEL: Record<string, string> = {
  v1: "Version 1",
  v2: "Version 2",
  incomplete_v2: "Incomplete V2",
  none: "No dataset yet",
};

export default function PersonPage({ params }: { params: Promise<{ person: string }> }) {
  const { person } = use(params);
  const sessions = useQuery({ queryKey: ["sessions", person], queryFn: () => api.sessions(person) });

  return (
    <div className="flex flex-col gap-6">
      <header>
        <p className="text-sm text-[var(--muted)]">
          <Link href="/people" className="underline">People</Link> / {person}
        </p>
        <h1 className="text-2xl font-semibold">{person}</h1>
        <p className="mt-2 text-sm">
          <Link href={`/people/${encodeURIComponent(person)}/compare`} className="underline">Compare sessions</Link>
        </p>
      </header>
      <Card title="Sessions">
        {sessions.isPending ? <Loading label="Loading sessions…" /> : null}
        {sessions.isError ? (
          <ErrorState message={sessions.error.message} code={(sessions.error as { code?: string }).code} />
        ) : null}
        {sessions.data && sessions.data.length === 0 ? (
          <Empty title="No sessions" detail="No recordings in this folder yet." />
        ) : null}
        {sessions.data && sessions.data.length > 0 ? (
          <table className="w-full text-left text-sm">
            <thead className="text-[var(--muted)]">
              <tr>
                <th scope="col" className="py-2 font-medium">Session</th>
                <th scope="col" className="py-2 font-medium">Video</th>
                <th scope="col" className="py-2 font-medium">Dataset</th>
              </tr>
            </thead>
            <tbody>
              {sessions.data.map((row) => (
                <tr key={row.session_id} className="border-t border-[var(--border)]">
                  <td className="py-2 font-mono">
                    {row.layout === "v1" || row.layout === "v2" ? (
                      <span className="flex flex-wrap gap-3">
                        {row.session_id}
                        <Link className="font-sans underline" href={`/people/${encodeURIComponent(person)}/sessions/${encodeURIComponent(row.session_id)}/facial`}>facial</Link>
                        <Link className="font-sans underline" href={`/people/${encodeURIComponent(person)}/sessions/${encodeURIComponent(row.session_id)}/diagnostics`}>diagnostics</Link>
                        <Link className="font-sans underline" href={`/people/${encodeURIComponent(person)}/sessions/${encodeURIComponent(row.session_id)}/3d`}>3D</Link>
                        {row.layout === "v2" ? (
                          <Link className="font-sans underline" href={`/people/${encodeURIComponent(person)}/sessions/${encodeURIComponent(row.session_id)}/body`}>body</Link>
                        ) : null}
                      </span>
                    ) : row.session_id}
                  </td>
                  <td className="py-2">
                    {row.has_video ? <Badge tone="good">present</Badge> : <Badge>missing</Badge>}
                  </td>
                  <td className="py-2">
                    <Badge tone={row.layout === "v2" ? "good" : row.layout === "incomplete_v2" ? "warn" : "neutral"}>
                      {LAYOUT_LABEL[row.layout] ?? row.layout}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </Card>
    </div>
  );
}
