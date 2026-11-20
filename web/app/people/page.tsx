"use client";

import Link from "next/link";
import { useMemo, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { useApiToken } from "../lib/token";
import { Card, Empty, ErrorState, Loading } from "../components/ui";
import { ConfirmDialog, TokenField, errorText } from "../components/manage-ui";

export default function PeoplePage() {
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState("");
  const [newName, setNewName] = useState("");
  const [token, saveToken, tokenLoaded] = useApiToken();
  const [deleting, setDeleting] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const persons = useQuery({ queryKey: ["persons"], queryFn: api.persons });
  const shown = useMemo(
    () => (persons.data ?? []).filter((p) => p.toLowerCase().includes(filter.trim().toLowerCase())),
    [persons.data, filter],
  );

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ["persons"] });
    queryClient.invalidateQueries({ queryKey: ["sessions"] });
  };

  const add = useMutation({
    mutationFn: () => api.createPerson(token, newName.trim()),
    onSuccess: (r) => { setNewName(""); setNotice(`Added ${r.person}.`); refresh(); },
  });

  const remove = useMutation({
    mutationFn: ({ token: t, person }: { token: string; person: string }) => api.deletePerson(t, person, person),
    onSuccess: (_r, v) => { setDeleting(null); setNotice(`Deleted everything for ${v.person}.`); refresh(); },
  });

  const onAdd = (event: FormEvent) => {
    event.preventDefault();
    if (!tokenLoaded || token === "" || newName.trim() === "") return;
    add.mutate();
  };

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">People</h1>
        <p className="text-sm text-[var(--muted)]">People and projects with recording folders.</p>
      </header>

      {notice ? <p role="status" className="text-sm font-medium">{notice}</p> : null}

      <Card title="Add a person">
        <form onSubmit={onAdd} className="grid gap-3 text-sm md:grid-cols-2">
          <label className="flex flex-col gap-1">
            Name (lowercase letters, numbers, - and _)
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono"
            />
          </label>
          <TokenField token={token} onChange={saveToken} />
          <div className="flex items-center gap-3 md:col-span-2">
            <button
              type="submit"
              disabled={!tokenLoaded || token === "" || newName.trim() === "" || add.isPending}
              className="rounded border border-[var(--border)] px-3 py-1.5 disabled:opacity-50"
            >
              Add person
            </button>
          </div>
          {add.isError ? <div className="md:col-span-2"><ErrorState {...errorText(add.error)} /></div> : null}
        </form>
      </Card>

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
        {persons.data && persons.data.length === 0 ? <Empty title="No people yet" detail="Add one above." /> : null}
        {persons.data && persons.data.length > 0 && shown.length === 0 ? <Empty title="No matches" /> : null}
        <ul className="divide-y divide-[var(--border)]">
          {shown.map((person) => (
            <li key={person} className="flex items-center justify-between gap-3">
              <Link href={`/people/${encodeURIComponent(person)}`} className="block flex-1 rounded px-2 py-3 text-sm hover:bg-slate-500/10 focus-visible:ring-2 focus-visible:ring-[var(--accent)] focus:outline-none">
                {person}
              </Link>
              <button
                type="button"
                onClick={() => setDeleting(person)}
                className="mr-2 rounded border border-red-300 px-2 py-1 text-xs text-red-800 dark:text-red-200"
              >
                Delete person
              </button>
            </li>
          ))}
        </ul>
      </Card>

      {deleting ? (
        <ConfirmDialog
          title="Delete this person?"
          expected={deleting}
          actionLabel="Delete person"
          pending={remove.isPending}
          error={remove.isError ? remove.error : undefined}
          onClose={() => { setDeleting(null); remove.reset(); }}
          onConfirm={(t) => remove.mutate({ token: t, person: deleting })}
        >
          <p>{deleting}: every recording and every dataset for this person is removed. This cannot be undone.</p>
        </ConfirmDialog>
      ) : null}
    </div>
  );
}
