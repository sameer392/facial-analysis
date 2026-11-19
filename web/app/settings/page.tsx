"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Card, ErrorState, Loading } from "../components/ui";

export default function SettingsPage() {
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settings });

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="text-sm text-[var(--muted)]">
          How this copy of the app is set up. Values are read from the running API; change them in the code or the
          environment, not here.
        </p>
      </header>

      <Card title="Connection and data">
        {settings.isPending ? <Loading /> : null}
        {settings.isError ? <ErrorState message={settings.error.message} /> : null}
        {settings.data ? (
          <dl className="grid gap-3 text-sm md:grid-cols-2">
            <div><dt className="text-[var(--muted)]">Data folder</dt><dd className="font-mono">{settings.data.data_root_name}</dd></div>
            <div><dt className="text-[var(--muted)]">Model folder</dt><dd className="font-mono">{settings.data.model_folder}</dd></div>
            <div><dt className="text-[var(--muted)]">API address</dt><dd className="font-mono">{settings.data.bind_address}</dd></div>
            <div><dt className="text-[var(--muted)]">API version</dt><dd>{settings.data.api_version}</dd></div>
            <div><dt className="text-[var(--muted)]">Dataset schema</dt><dd>{settings.data.schema_version}</dd></div>
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
            <div>
              <dt className="text-[var(--muted)]">API token</dt>
              <dd>
                {settings.data.api_token_configured ? "Configured on the server" : "Not configured"}. Sent in the{" "}
                <span className="font-mono">{settings.data.token_header}</span> header. It is entered on the Jobs page for
                this browser tab and is never shown here.
              </dd>
            </div>
          </dl>
        ) : null}
      </Card>
    </div>
  );
}
