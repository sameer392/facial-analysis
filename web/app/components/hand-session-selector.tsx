"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";

/** Choose a person and a session. Only sessions with a recording are offered; pose data is needed for hands. */
export function HandSessionSelector({
  person,
  session,
  onPerson,
  onSession,
}: {
  person: string;
  session: string;
  onPerson: (value: string) => void;
  onSession: (value: string) => void;
}) {
  const persons = useQuery({ queryKey: ["persons"], queryFn: api.persons });
  const sessions = useQuery({
    queryKey: ["sessions", person],
    queryFn: () => api.sessions(person),
    enabled: person !== "",
  });
  const rows = (sessions.data ?? []).filter((s) => s.has_video);

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <label className="flex flex-col gap-1 text-sm">
        Person
        <select
          value={person}
          onChange={(e) => {
            onPerson(e.target.value);
            onSession("");
          }}
          className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
        >
          <option value="">Choose a person</option>
          {(persons.data ?? []).map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-1 text-sm">
        Recording (session)
        <select
          value={session}
          onChange={(e) => onSession(e.target.value)}
          disabled={person === "" || rows.length === 0}
          className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
        >
          <option value="">{person === "" ? "Choose a person first" : rows.length === 0 ? "No recordings" : "Choose a session"}</option>
          {rows.map((s) => (
            <option key={s.session_id} value={s.session_id}>
              {s.session_id}{s.layout === "pose" ? "" : " (pose data not built)"}
            </option>
          ))}
        </select>
      </label>
      {sessions.isError ? <p className="text-sm text-red-700">The sessions could not be loaded.</p> : null}
    </div>
  );
}
