"use client";

import { useQuery } from "@tanstack/react-query";
import { api, type LayerStatus } from "../lib/api";
import { Badge, Card, ErrorState, Loading } from "./ui";

/** A page for a processing layer. Until its extractor exists, it says so and states what is needed. */
export function LayerPage({ layerKey, heading }: { layerKey: LayerStatus["key"]; heading: string }) {
  const layers = useQuery({ queryKey: ["layers"], queryFn: api.layers });
  const layer = layers.data?.find((l) => l.key === layerKey);

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-semibold">{heading}</h1>
      </header>
      {layers.isPending ? <Loading label="Checking layer status…" /> : null}
      {layers.isError ? <ErrorState message={layers.error.message} /> : null}
      {layer ? (
        <Card title="Status">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            {layer.implemented ? (
              <Badge tone="good">Extractor present</Badge>
            ) : (
              <Badge>Not implemented</Badge>
            )}
            <span className="text-[var(--muted)]">{layer.extractor}</span>
          </div>
          <p className="mt-4 text-sm">{layer.implemented
            ? "The extractor file exists. Its results are not shown on this page yet."
            : "This page is a placeholder. No data is shown because the extractor has not been written."}</p>
          <dl className="mt-4 grid gap-3 text-sm">
            <div>
              <dt className="text-[var(--muted)]">Will show</dt>
              <dd>{layer.will_show}</dd>
            </div>
            <div>
              <dt className="text-[var(--muted)]">Needed next</dt>
              <dd>{layer.needed}</dd>
            </div>
          </dl>
        </Card>
      ) : null}
    </div>
  );
}
