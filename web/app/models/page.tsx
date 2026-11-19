"use client";

import { useQuery } from "@tanstack/react-query";
import { api, type ModelStatus } from "../lib/api";
import { Badge, Card, Empty, ErrorState, Loading } from "../components/ui";

const STATUS_LABEL: Record<ModelStatus["sha256_status"], string> = {
  match: "Hash matches",
  mismatch: "Hash does not match",
  missing: "File missing",
};

function statusTone(status: ModelStatus["sha256_status"]): "good" | "warn" | "neutral" {
  if (status === "match") return "good";
  if (status === "mismatch") return "warn";
  return "neutral";
}

export default function ModelsPage() {
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">Models</h1>
        <p className="text-sm text-[var(--muted)]">
          The MediaPipe model files the analysis uses. Each file is hashed and compared with its recorded SHA-256.
          Models are never loaded here. Licence text is shown as recorded; it is evidence, not clearance.
        </p>
      </header>

      {models.isPending ? <Loading label="Checking model files…" /> : null}
      {models.isError ? <ErrorState message={models.error.message} /> : null}
      {models.data && models.data.length === 0 ? <Empty title="No models registered" /> : null}

      {models.data?.map((m) => (
        <Card key={m.key} title={m.filename}>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <Badge tone={statusTone(m.sha256_status)}>{STATUS_LABEL[m.sha256_status]}</Badge>
            <span className="text-[var(--muted)]">{m.location}</span>
            {m.size_bytes != null ? <span className="text-[var(--muted)]">{(m.size_bytes / 1_048_576).toFixed(1)} MB</span> : null}
          </div>
          <dl className="mt-4 grid gap-3 text-sm md:grid-cols-2">
            <div>
              <dt className="text-[var(--muted)]">Recorded SHA-256</dt>
              <dd className="break-all font-mono text-xs">{m.expected_sha256}</dd>
            </div>
            <div>
              <dt className="text-[var(--muted)]">Actual SHA-256</dt>
              <dd className="break-all font-mono text-xs">{m.actual_sha256 ?? "not available"}</dd>
            </div>
            <div>
              <dt className="text-[var(--muted)]">Version</dt>
              <dd>{m.version}</dd>
            </div>
            <div>
              <dt className="text-[var(--muted)]">Source</dt>
              <dd className="break-all">{m.source_url}</dd>
            </div>
            <div className="md:col-span-2">
              <dt className="text-[var(--muted)]">Licence status (as recorded)</dt>
              <dd>{m.licence_status}</dd>
            </div>
          </dl>
        </Card>
      ))}
    </div>
  );
}
