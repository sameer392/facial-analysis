"use client";

import { useMemo, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { useApiToken } from "../lib/token";
import { Card, ErrorState } from "./ui";
import { NeedsToken, errorText } from "./manage-ui";

/**
 * Adds a recording to a person. With fixedPerson it is used on that person's page; without it, the page picks
 * the person. The suggested session name is the next free one for that person.
 */
export function UploadRecording({ fixedPerson }: { fixedPerson?: string }) {
  const queryClient = useQueryClient();
  const [token, , tokenLoaded] = useApiToken();
  const [chosen, setChosen] = useState("");
  const [session, setSession] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const persons = useQuery({ queryKey: ["persons"], queryFn: api.persons, enabled: !fixedPerson });
  const person = fixedPerson ?? chosen;
  const sessions = useQuery({
    queryKey: ["sessions", person],
    queryFn: () => api.sessions(person),
    enabled: person !== "",
  });
  const suggested = useMemo(() => {
    const used = (sessions.data ?? []).map((s) => Number(s.session_id.replace("session-", ""))).filter(Number.isFinite);
    return `session-${String((used.length ? Math.max(...used) : 0) + 1).padStart(3, "0")}`;
  }, [sessions.data]);

  const upload = useMutation({
    mutationFn: () => api.uploadVideo(token, person, session || suggested, file as File, setProgress),
    onSettled: () => setProgress(null),
    onSuccess: (r) => {
      setFile(null);
      setSession("");
      setNotice(`Uploaded ${r.person} / ${r.session} (${(r.bytes / 1_048_576).toFixed(1)} MB). Run Face data on the Jobs page next.`);
      queryClient.invalidateQueries({ queryKey: ["sessions", r.person] });
      queryClient.invalidateQueries({ queryKey: ["persons"] });
    },
  });

  const onUpload = (event: FormEvent) => {
    event.preventDefault();
    if (!tokenLoaded || token === "" || person === "" || !file) return;
    upload.mutate();
  };

  return (
    <Card title="Add a recording">
      <form onSubmit={onUpload} className="grid gap-3 text-sm md:grid-cols-2">
        {!fixedPerson ? (
          <label className="flex flex-col gap-1">
            Person
            <select value={chosen} onChange={(e) => { setChosen(e.target.value); setSession(""); }}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1">
              <option value="">Choose a person</option>
              {(persons.data ?? []).map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
        ) : null}
        <label className="flex flex-col gap-1">
          Session (suggested: {person === "" ? "choose a person" : suggested})
          <input
            value={session}
            placeholder={suggested}
            onChange={(e) => setSession(e.target.value)}
            disabled={person === ""}
            className="rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono"
          />
        </label>
        <label className="flex flex-col gap-1">
          MP4 video (at most 2 GB)
          <input type="file" accept="video/mp4" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="text-xs" />
        </label>
        <div className="md:col-span-2">
          <NeedsToken token={token} loaded={tokenLoaded} />
        </div>
        {progress !== null ? (
          <div className="md:col-span-2">
            <progress value={progress} max={1} className="w-full" aria-label="Upload progress" />
            <p className="text-xs text-[var(--muted)]">{Math.round(progress * 100)} % sent</p>
          </div>
        ) : null}
        <div className="flex flex-wrap items-center gap-3 md:col-span-2">
          <button
            type="submit"
            disabled={!tokenLoaded || token === "" || person === "" || !file || upload.isPending}
            className="rounded border border-[var(--border)] px-3 py-1.5 disabled:opacity-50"
          >
            {upload.isPending ? "Uploading…" : "Upload"}
          </button>
          <p className="text-xs text-[var(--muted)]">An existing session name is refused; nothing is overwritten.</p>
        </div>
        {notice ? <p role="status" className="text-sm font-medium md:col-span-2">{notice}</p> : null}
        {upload.isError ? <div className="md:col-span-2"><ErrorState {...errorText(upload.error)} /></div> : null}
      </form>
    </Card>
  );
}
