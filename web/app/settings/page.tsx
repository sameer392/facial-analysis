"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { useApiToken } from "../lib/token";
import { Card, ErrorState, Loading } from "../components/ui";

export default function SettingsPage() {
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settings });
  const [token, saveToken, loaded] = useApiToken();
  const [draft, setDraft] = useState("");
  const [check, setCheck] = useState<"idle" | "checking" | "ok" | "wrong" | "unreachable">("idle");

  const save = () => {
    saveToken(draft);
    setDraft("");
    setCheck("idle");
  };
  const test = async () => {
    setCheck("checking");
    setCheck(await api.checkToken(token));
  };

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="text-sm text-[var(--muted)]">
          How this copy of the app is set up. The API token is set here once; the other pages use it.
        </p>
      </header>

      <Card title="API token">
        <p className="mb-3 text-sm">
          The token is in <code className="font-mono">config/api_token.txt</code> in the project folder. It is created
          the first time the API starts and stays the same across restarts. Paste it here once. It is kept in this
          browser only.
        </p>
        <div className="flex flex-col gap-3 text-sm">
          <p>
            Status:{" "}
            {!loaded ? "…" : token !== "" ? <span className="font-medium">saved in this browser</span> : <span className="font-medium">not set</span>}
          </p>
          <label className="flex flex-col gap-1">
            Token
            <input
              type="password"
              autoComplete="off"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={token !== "" ? "saved (type to replace)" : "paste the token"}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono"
            />
          </label>
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" onClick={save} disabled={draft.trim() === ""}
              className="rounded border border-[var(--border)] px-3 py-1.5 disabled:opacity-40">Save token</button>
            <button type="button" onClick={test} disabled={token === "" || check === "checking"}
              className="rounded border border-[var(--border)] px-3 py-1.5 disabled:opacity-40">Check token</button>
            <button type="button" onClick={() => { saveToken(""); setCheck("idle"); }} disabled={token === ""}
              className="rounded border border-[var(--border)] px-3 py-1.5 disabled:opacity-40">Forget token</button>
          </div>
          {check === "ok" ? <p className="font-medium text-emerald-800 dark:text-emerald-200">The token is accepted.</p> : null}
          {check === "wrong" ? <p className="font-medium text-red-800 dark:text-red-200">The API refused this token. Paste the value from config/api_token.txt.</p> : null}
          {check === "unreachable" ? <p className="font-medium text-amber-800 dark:text-amber-200">The API could not be reached. Is it running?</p> : null}
        </div>
      </Card>

      <Card title="Connection and data">
        {settings.isPending ? <Loading label="Loading settings…" /> : null}
        {settings.isError ? <ErrorState message={settings.error.message} /> : null}
        {settings.data ? (
          <dl className="grid gap-3 text-sm md:grid-cols-2">
            <div><dt className="text-[var(--muted)]">Data folder</dt><dd className="font-mono">{settings.data.data_root_name}</dd></div>
            <div><dt className="text-[var(--muted)]">Model folder</dt><dd className="font-mono">{settings.data.model_folder}</dd></div>
            <div><dt className="text-[var(--muted)]">API address</dt><dd className="font-mono">{settings.data.bind_address}</dd></div>
            <div><dt className="text-[var(--muted)]">API version</dt><dd>{settings.data.api_version}</dd></div>
            <div><dt className="text-[var(--muted)]">Python</dt><dd>{settings.data.python_version} ({settings.data.platform})</dd></div>
          </dl>
        ) : null}
      </Card>

      <Card title="Writes and access">
        {settings.data ? (
          <dl className="grid gap-3 text-sm">
            <div>
              <dt className="text-[var(--muted)]">Writes</dt>
              <dd>{settings.data.writes}</dd>
            </div>
          </dl>
        ) : null}
      </Card>
    </div>
  );
}
