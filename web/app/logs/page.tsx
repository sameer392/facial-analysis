"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ACTIVE_JOB_STATES, api } from "../lib/api";
import { Badge, Card, Empty, ErrorState, Loading } from "../components/ui";

export default function LogsPage() {
  const [selected, setSelected] = useState<string | null>(null);
  const logs = useQuery({
    queryKey: ["job-logs"],
    queryFn: () => api.jobLogs(),
    refetchInterval: (query) => (query.state.data?.some((j) => ACTIVE_JOB_STATES.includes(j.state)) ? 3000 : false),
  });
  const tail = useQuery({
    queryKey: ["job-log", selected],
    queryFn: () => api.jobLogTail(selected as string, 300),
    enabled: selected !== null,
    refetchInterval: (query) => {
      const row = logs.data?.find((j) => j.job_id === selected);
      return row && ACTIVE_JOB_STATES.includes(row.state) ? 3000 : false;
    },
  });

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">Logs</h1>
        <p className="text-sm text-[var(--muted)]">
          Output from the job workers, with the project and home folders shown as placeholders. The API does not
          write a separate log yet. Tokens are never logged.
        </p>
      </header>

      <Card title="Job logs">
        {logs.isPending ? <Loading label="Loading job logs…" /> : null}
        {logs.isError ? <ErrorState message={logs.error.message} /> : null}
        {logs.data && logs.data.length === 0 ? <Empty title="No job logs yet" detail="Logs appear after a job is started." /> : null}
        {logs.data && logs.data.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-[var(--muted)]">
                <tr>
                  <th scope="col" className="py-2 font-medium">Started</th>
                  <th scope="col" className="py-2 font-medium">Job</th>
                  <th scope="col" className="py-2 font-medium">Recording</th>
                  <th scope="col" className="py-2 font-medium">State</th>
                  <th scope="col" className="py-2 font-medium"><span className="sr-only">Open</span></th>
                </tr>
              </thead>
              <tbody>
                {logs.data.map((row) => (
                  <tr key={row.job_id} className="border-t border-[var(--border)]">
                    <td className="py-2 whitespace-nowrap">{new Date(row.created_at).toLocaleString()}</td>
                    <td className="py-2">{row.kind}</td>
                    <td className="py-2 font-mono">{row.person} / {row.session}</td>
                    <td className="py-2"><Badge>{row.state}</Badge></td>
                    <td className="py-2">
                      <button
                        type="button"
                        onClick={() => setSelected(row.job_id)}
                        aria-pressed={selected === row.job_id}
                        className="rounded border border-[var(--border)] px-2 py-1"
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </Card>

      {selected ? (
        <Card title={`Last lines of ${selected}`}>
          {tail.isPending ? <Loading /> : null}
          {tail.isError ? <ErrorState message={tail.error.message} /> : null}
          {tail.data ? (
            <>
              <p className="mb-2 text-xs text-[var(--muted)]">
                {tail.data.total_lines} lines in total; showing the last {tail.data.lines.length}.
              </p>
              <pre className="max-h-[480px] overflow-auto rounded bg-slate-900 p-3 text-xs leading-relaxed text-slate-100">
                {tail.data.lines.join("\n")}
              </pre>
            </>
          ) : null}
        </Card>
      ) : null}
    </div>
  );
}
