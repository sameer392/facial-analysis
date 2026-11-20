"use client";

import { use, useMemo, useState, type FormEvent } from "react";
import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../lib/api";
import { useApiToken } from "../../lib/token";
import { Badge, Card, Empty, ErrorState, Loading } from "../../components/ui";
import { ConfirmDialog, RenameDialog, TokenField, errorText } from "../../components/manage-ui";

type Target = { kind: "data" | "video"; session: string };

export default function PersonPage({ params }: { params: Promise<{ person: string }> }) {
  const { person } = use(params);
  const queryClient = useQueryClient();
  const [token, saveToken, tokenLoaded] = useApiToken();
  const sessions = useQuery({ queryKey: ["sessions", person], queryFn: () => api.sessions(person) });
  const [file, setFile] = useState<File | null>(null);
  const [session, setSession] = useState("");
  const [target, setTarget] = useState<Target | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const rename = useMutation({
    mutationFn: ({ token: t, from, to }: { token: string; from: string; to: string }) => api.renameSession(t, person, from, to),
    onSuccess: (_r, v) => { setRenaming(null); setNotice(`Renamed ${v.from} to ${v.to}.`); refresh(); },
  });
  const [notice, setNotice] = useState<string | null>(null);

  const suggested = useMemo(() => {
    const used = (sessions.data ?? []).map((s) => Number(s.session_id.replace("session-", ""))).filter(Number.isFinite);
    return `session-${String((used.length ? Math.max(...used) : 0) + 1).padStart(3, "0")}`;
  }, [sessions.data]);

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ["sessions", person] });
    queryClient.invalidateQueries({ queryKey: ["persons"] });
  };

  const upload = useMutation({
    mutationFn: () => api.uploadVideo(token, person, session || suggested, file as File),
    onSuccess: (r) => {
      setFile(null);
      setSession("");
      setNotice(`Uploaded ${r.session} (${(r.bytes / 1_048_576).toFixed(1)} MB). Run Face data on the Jobs page next.`);
      refresh();
    },
  });

  const remove = useMutation({
    mutationFn: ({ token: t, target: tg }: { token: string; target: Target }) =>
      tg.kind === "data"
        ? api.deleteDataset(t, person, tg.session, `${person}/${tg.session}`)
        : api.deleteVideo(t, person, tg.session, `${person}/${tg.session}`),
    onSuccess: (_r, v) => {
      setTarget(null);
      setNotice(v.target.kind === "data" ? `Deleted the data for ${v.target.session}.` : `Deleted the video for ${v.target.session}.`);
      refresh();
    },
  });

  const onUpload = (event: FormEvent) => {
    event.preventDefault();
    if (!tokenLoaded || token === "") return;
    upload.mutate();
  };

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

      {notice ? <p role="status" className="text-sm font-medium">{notice}</p> : null}

      <Card title="Add a recording">
        <form onSubmit={onUpload} className="grid gap-3 text-sm md:grid-cols-2">
          <label className="flex flex-col gap-1">
            Session (suggested: {suggested})
            <input
              value={session}
              placeholder={suggested}
              onChange={(e) => setSession(e.target.value)}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono"
            />
          </label>
          <label className="flex flex-col gap-1">
            MP4 video (at most 2 GB)
            <input type="file" accept="video/mp4" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="text-xs" />
          </label>
          <div className="md:col-span-2">
            <TokenField token={token} onChange={saveToken} />
          </div>
          <div className="flex items-center gap-3 md:col-span-2">
            <button
              type="submit"
              disabled={!tokenLoaded || token === "" || !file || upload.isPending}
              className="rounded border border-[var(--border)] px-3 py-1.5 disabled:opacity-50"
            >
              {upload.isPending ? "Uploading…" : "Upload"}
            </button>
            <p className="text-xs text-[var(--muted)]">An existing session name is refused; nothing is overwritten.</p>
          </div>
          {upload.isError ? <div className="md:col-span-2"><ErrorState {...errorText(upload.error)} /></div> : null}
        </form>
      </Card>

      <Card title="Sessions">
        {sessions.isPending ? <Loading label="Loading sessions…" /> : null}
        {sessions.isError ? (
          <ErrorState message={sessions.error.message} code={(sessions.error as { code?: string }).code} />
        ) : null}
        {sessions.data && sessions.data.length === 0 ? (
          <Empty title="No sessions" detail="Upload a recording above." />
        ) : null}
        {sessions.data && sessions.data.length > 0 ? (
          <table className="w-full text-left text-sm">
            <thead className="text-[var(--muted)]">
              <tr>
                <th scope="col" className="py-2 font-medium">Session</th>
                <th scope="col" className="py-2 font-medium">Video</th>
                <th scope="col" className="py-2 font-medium">Data</th>
                <th scope="col" className="py-2 font-medium"><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {sessions.data.map((row) => (
                <tr key={row.session_id} className="border-t border-[var(--border)] align-top">
                  <td className="py-2 font-mono">
                    {row.layout === "face" || row.layout === "pose" ? (
                      <span className="flex flex-wrap gap-3">
                        {row.session_id}
                        <Link className="font-sans underline" href={`/people/${encodeURIComponent(person)}/sessions/${encodeURIComponent(row.session_id)}/facial`}>facial</Link>
                        <Link className="font-sans underline" href={`/people/${encodeURIComponent(person)}/sessions/${encodeURIComponent(row.session_id)}/diagnostics`}>diagnostics</Link>
                        <Link className="font-sans underline" href={`/people/${encodeURIComponent(person)}/sessions/${encodeURIComponent(row.session_id)}/3d`}>3D</Link>
                        {row.layout === "pose" ? (
                          <Link className="font-sans underline" href={`/people/${encodeURIComponent(person)}/sessions/${encodeURIComponent(row.session_id)}/body`}>body</Link>
                        ) : null}
                      </span>
                    ) : row.session_id}
                  </td>
                  <td className="py-2">
                    {row.has_video ? <Badge tone="good">present</Badge> : <Badge>missing</Badge>}
                  </td>
                  <td className="py-2">
                    <Badge tone={row.layout === "pose" ? "good" : row.layout === "pose_incomplete" ? "warn" : "neutral"}>
                      {row.layout_name}
                    </Badge>
                  </td>
                  <td className="py-2">
                    <span className="flex flex-wrap justify-end gap-2">
                      {row.layout !== "none" ? (
                        <button type="button" onClick={() => setTarget({ kind: "data", session: row.session_id })}
                          className="rounded border border-[var(--border)] px-2 py-1 text-xs">Delete data</button>
                      ) : null}
                      <button type="button" onClick={() => setRenaming(row.session_id)}
                        className="rounded border border-[var(--border)] px-2 py-1 text-xs">Rename</button>
                      {row.has_video ? (
                        <button type="button" onClick={() => setTarget({ kind: "video", session: row.session_id })}
                          className="rounded border border-red-300 px-2 py-1 text-xs text-red-800 dark:text-red-200">Delete video</button>
                      ) : null}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </Card>

      {renaming ? (
        <RenameDialog
          title="Rename this recording?"
          current={renaming}
          pending={rename.isPending}
          error={rename.isError ? rename.error : undefined}
          onClose={() => { setRenaming(null); rename.reset(); }}
          onConfirm={(t, to) => rename.mutate({ token: t, from: renaming, to })}
        />
      ) : null}
      {target ? (
        <ConfirmDialog
          title={target.kind === "data" ? "Delete the processed data?" : "Delete the video?"}
          expected={`${person}/${target.session}`}
          actionLabel={target.kind === "data" ? "Delete data" : "Delete video"}
          pending={remove.isPending}
          error={remove.isError ? remove.error : undefined}
          onClose={() => { setTarget(null); remove.reset(); }}
          onConfirm={(t) => remove.mutate({ token: t, target })}
        >
          <p>
            {person} / {target.session}.{" "}
            {target.kind === "data"
              ? "The face and pose data is removed. The video is kept."
              : "The video is removed. The processed data is kept."}
          </p>
        </ConfirmDialog>
      ) : null}
    </div>
  );
}
