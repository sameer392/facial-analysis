"use client";

import { useState, type ReactNode } from "react";
import { ApiFailure } from "../lib/api";
import { useApiToken } from "../lib/token";
import { ErrorState } from "./ui";

export function errorText(error: unknown): { message: string; code?: string } {
  if (error instanceof ApiFailure) return { message: error.message, code: error.code };
  if (error instanceof Error) return { message: error.message };
  return { message: "Unknown error" };
}

/** The API token, kept for this browser tab. Shared by every page that changes data. */
export function TokenField({ token, onChange }: { token: string; onChange: (value: string) => void }) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      API token (kept for this tab only)
      <input
        type="password"
        autoComplete="off"
        value={token}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono"
      />
    </label>
  );
}

/**
 * Asks for the exact name before a deletion, and for the token if none is set. The parent unmounts this
 * dialog when it closes, so the typed text is cleared each time.
 */
export function ConfirmDialog({
  title, children, expected, actionLabel, pending = false, error, onClose, onConfirm,
}: {
  title: string;
  children?: ReactNode;
  expected: string;
  actionLabel: string;
  pending?: boolean;
  error?: unknown;
  onClose: () => void;
  onConfirm: (token: string, typed: string) => void;
}) {
  const [token, saveToken, loaded] = useApiToken();
  const [typed, setTyped] = useState("");
  const ready = loaded && token !== "" && typed === expected && !pending;
  return (
    <div role="dialog" aria-modal="true" aria-labelledby="confirm-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onKeyDown={(e) => { if (e.key === "Escape") onClose(); }}>
      <div className="w-full max-w-md rounded-lg border border-[var(--border)] bg-white p-5 shadow-lg dark:bg-slate-900">
        <h2 id="confirm-title" className="text-lg font-semibold">{title}</h2>
        <div className="mt-2 text-sm text-[var(--muted)]">{children}</div>
        <div className="mt-4 flex flex-col gap-3">
          <TokenField token={token} onChange={saveToken} />
          <label className="flex flex-col gap-1 text-sm">
            Type <code className="font-mono">{expected}</code> to confirm
            <input
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              autoFocus
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono"
            />
          </label>
          {error ? <ErrorState {...errorText(error)} /> : null}
        </div>
        <div className="mt-5 flex justify-end gap-2 text-sm">
          <button type="button" onClick={onClose} className="rounded border border-[var(--border)] px-3 py-1.5">Cancel</button>
          <button
            type="button"
            disabled={!ready}
            onClick={() => onConfirm(token, typed)}
            className="rounded border border-red-300 px-3 py-1.5 text-red-800 disabled:opacity-40 dark:text-red-200"
          >
            {pending ? "Working…" : actionLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
