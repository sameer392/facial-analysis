"use client";

import { useMemo, useState, type FormEvent } from "react";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiFailure, api, type SessionRow } from "../lib/api";
import { useApiToken } from "../lib/token";
import { Badge, Card, Empty, ErrorState, Loading } from "../components/ui";

const PHRASE = "clear job history";

function errorText(error: unknown): { message: string; code?: string } {
  if (error instanceof ApiFailure) return { message: error.message, code: error.code };
  if (error instanceof Error) return { message: error.message };
  return { message: "Unknown error" };
}

/** A delete button that stays disabled until the exact name is typed. */
function TypedDelete({ expected, label, disabled, onConfirm }: {
  expected: string;
  label: string;
  disabled: boolean;
  onConfirm: () => void;
}) {
  const [typed, setTyped] = useState("");
  const ready = !disabled && typed === expected;
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <input
        value={typed}
        onChange={(e) => setTyped(e.target.value)}
        placeholder={expected}
        aria-label={`Type ${expected} to confirm`}
        className="w-64 rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono text-xs"
      />
      <button
        type="button"
        disabled={!ready}
        onClick={() => { onConfirm(); setTyped(""); }}
        className="rounded border border-red-300 px-2 py-1 text-red-800 disabled:opacity-40"
      >
        {label}
      </button>
    </div>
  );
}

export default function ManagePage() {
  const queryClient = useQueryClient();
  const [token, saveToken, tokenLoaded] = useApiToken();
  const [newPerson, setNewPerson] = useState("");
  const [uploadPerson, setUploadPerson] = useState("");
  const [uploadSession, setUploadSession] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const persons = useQuery({ queryKey: ["persons"], queryFn: api.persons });
  const sessionQueries = useQueries({
    queries: (persons.data ?? []).map((person) => ({
      queryKey: ["sessions", person],
      queryFn: () => api.sessions(person),
    })),
  });
  const rows = useMemo(
    () => (persons.data ?? []).flatMap((person, i) =>
      (sessionQueries[i]?.data ?? []).map((s: SessionRow) => ({ person, ...s }))),
    [persons.data, sessionQueries],
  );
  const sessionsOfUploadPerson = rows.filter((r) => r.person === uploadPerson);
  const suggestedSession = useMemo(() => {
    const used = sessionsOfUploadPerson.map((r) => Number(r.session_id.replace("session-", ""))).filter(Number.isFinite);
    return `session-${String((used.length ? Math.max(...used) : 0) + 1).padStart(3, "0")}`;
  }, [sessionsOfUploadPerson]);

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ["persons"] });
    queryClient.invalidateQueries({ queryKey: ["sessions"] });
    queryClient.invalidateQueries({ queryKey: ["jobs"] });
  };
  const done = (message: string) => { setNotice(message); refresh(); };

  const createPerson = useMutation({
    mutationFn: () => api.createPerson(token, newPerson.trim()),
    onSuccess: (r) => { setNewPerson(""); done(`Created ${r.person}.`); },
  });
  const upload = useMutation({
    mutationFn: () => api.uploadVideo(token, uploadPerson, uploadSession || suggestedSession, file as File),
    onSuccess: (r) => { setFile(null); done(`Uploaded ${r.person} / ${r.session} (${(r.bytes / 1_048_576).toFixed(1)} MB).`); },
  });
  const deleteDataset = useMutation({
    mutationFn: (r: { person: string; session: string }) =>
      api.deleteDataset(token, r.person, r.session, `${r.person}/${r.session}`),
    onSuccess: (_r, v) => done(`Deleted the data for ${v.person} / ${v.session}.`),
  });
  const deleteVideo = useMutation({
    mutationFn: (r: { person: string; session: string }) =>
      api.deleteVideo(token, r.person, r.session, `${r.person}/${r.session}`),
    onSuccess: (_r, v) => done(`Deleted the video ${v.person} / ${v.session}.`),
  });
  const deletePerson = useMutation({
    mutationFn: (person: string) => api.deletePerson(token, person, person),
    onSuccess: (_r, person) => done(`Deleted everything for ${person}.`),
  });
  const clearHistory = useMutation({
    mutationFn: () => api.clearJobHistory(token, PHRASE),
    onSuccess: (r) => done(`Removed ${r.removed} finished job record(s).`),
  });

  const tokenSet = tokenLoaded && token !== "";
  const failure = [createPerson, upload, deleteDataset, deleteVideo, deletePerson, clearHistory]
    .find((m) => m.isError);

  const onUpload = (event: FormEvent) => {
    event.preventDefault();
    upload.mutate();
  };

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">Manage</h1>
        <p className="text-sm text-[var(--muted)]">
          Add people and recordings, and remove them. Every action needs the API token. Deletions are permanent,
          and you must type the exact name to confirm.
        </p>
      </header>

      <Card title="API token">
        <label className="flex flex-col gap-1 text-sm">
          Token from the API console (kept for this tab only)
          <input
            type="password"
            autoComplete="off"
            value={token}
            onChange={(e) => saveToken(e.target.value)}
            className="rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono"
          />
        </label>
        {!tokenSet ? <p className="mt-2 text-xs text-[var(--muted)]">No token set. Every action on this page is disabled.</p> : null}
      </Card>

      {notice ? <p role="status" className="text-sm font-medium">{notice}</p> : null}
      {failure?.error ? <ErrorState {...errorText(failure.error)} /> : null}

      <Card title="Add a person">
        <form
          className="flex flex-wrap items-end gap-2 text-sm"
          onSubmit={(e) => { e.preventDefault(); createPerson.mutate(); }}
        >
          <label className="flex flex-col gap-1">
            Name (lowercase letters, numbers, - and _)
            <input
              value={newPerson}
              onChange={(e) => setNewPerson(e.target.value)}
              className="w-64 rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono"
            />
          </label>
          <button
            type="submit"
            disabled={!tokenSet || newPerson.trim() === "" || createPerson.isPending}
            className="rounded border border-[var(--border)] px-3 py-1.5 disabled:opacity-50"
          >
            Add person
          </button>
        </form>
      </Card>

      <Card title="Upload a recording">
        <form onSubmit={onUpload} className="grid gap-3 text-sm md:grid-cols-4">
          <label className="flex flex-col gap-1">
            Person
            <select
              value={uploadPerson}
              onChange={(e) => { setUploadPerson(e.target.value); setUploadSession(""); }}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
            >
              <option value="">Choose a person</option>
              {(persons.data ?? []).map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            Session (suggested: {suggestedSession})
            <input
              value={uploadSession}
              placeholder={suggestedSession}
              onChange={(e) => setUploadSession(e.target.value)}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono"
            />
          </label>
          <label className="flex flex-col gap-1">
            MP4 video
            <input
              type="file"
              accept="video/mp4"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="text-xs"
            />
          </label>
          <div className="flex items-end">
            <button
              type="submit"
              disabled={!tokenSet || uploadPerson === "" || !file || upload.isPending}
              className="rounded border border-[var(--border)] px-3 py-1.5 disabled:opacity-50"
            >
              {upload.isPending ? "Uploading…" : "Upload"}
            </button>
          </div>
        </form>
        <p className="mt-3 text-xs text-[var(--muted)]">
          Files must be MP4 and at most 2 GB. An existing session name is refused; nothing is overwritten.
          Uploading does not start any processing: run Face data and then Pose data on the Jobs page.
        </p>
      </Card>

      <Card title="Recordings">
        {persons.isPending ? <Loading /> : null}
        {rows.length === 0 && persons.isSuccess ? <Empty title="No recordings yet" detail="Upload one above." /> : null}
        {rows.length > 0 ? (
          <ul className="flex flex-col divide-y divide-[var(--border)] text-sm">
            {rows.map((r) => (
              <li key={`${r.person}/${r.session_id}`} className="flex flex-col gap-2 py-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-mono">{r.person} / {r.session_id}</span>
                  <span className="flex gap-2">
                    {r.has_video ? <Badge tone="good">video</Badge> : <Badge>no video</Badge>}
                    <Badge tone={r.layout === "none" ? "neutral" : "good"}>{r.layout_name}</Badge>
                  </span>
                </div>
                <div className="flex flex-col gap-2 md:flex-row md:flex-wrap md:gap-6">
                  {r.layout !== "none" ? (
                    <div>
                      <p className="mb-1 text-xs text-[var(--muted)]">Delete the processed data (the video is kept)</p>
                      <TypedDelete
                        expected={`${r.person}/${r.session_id}`}
                        label="Delete data"
                        disabled={!tokenSet || deleteDataset.isPending}
                        onConfirm={() => deleteDataset.mutate({ person: r.person, session: r.session_id })}
                      />
                    </div>
                  ) : null}
                  {r.has_video ? (
                    <div>
                      <p className="mb-1 text-xs text-[var(--muted)]">Delete the video</p>
                      <TypedDelete
                        expected={`${r.person}/${r.session_id}`}
                        label="Delete video"
                        disabled={!tokenSet || deleteVideo.isPending}
                        onConfirm={() => deleteVideo.mutate({ person: r.person, session: r.session_id })}
                      />
                    </div>
                  ) : null}
                </div>
              </li>
            ))}
          </ul>
        ) : null}
      </Card>

      <Card title="Delete a person">
        <p className="mb-2 text-xs text-[var(--muted)]">Removes every recording and dataset for the person.</p>
        {(persons.data ?? []).length === 0 ? <p className="text-sm text-[var(--muted)]">No people yet.</p> : null}
        <ul className="flex flex-col gap-3">
          {(persons.data ?? []).map((p) => (
            <li key={p}>
              <p className="mb-1 font-mono text-sm">{p}</p>
              <TypedDelete
                expected={p}
                label="Delete person"
                disabled={!tokenSet || deletePerson.isPending}
                onConfirm={() => deletePerson.mutate(p)}
              />
            </li>
          ))}
        </ul>
      </Card>

      <Card title="Job history">
        <p className="mb-2 text-xs text-[var(--muted)]">
          Removes finished job records and their logs. Jobs that are still running are kept.
        </p>
        <TypedDelete
          expected={PHRASE}
          label="Clear finished jobs"
          disabled={!tokenSet || clearHistory.isPending}
          onConfirm={() => clearHistory.mutate()}
        />
      </Card>
    </div>
  );
}
