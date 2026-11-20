"use client";

import Link from "next/link";
import { useState, type ReactNode } from "react";
import { ApiFailure } from "../lib/api";
import { useApiToken } from "../lib/token";
import { ErrorState } from "./ui";

export function errorText(error: unknown): { message: string; code?: string } {
  if (error instanceof ApiFailure) return { message: error.message, code: error.code };
  if (error instanceof Error) return { message: error.message };
  return { message: "Unknown error" };
}

/** Shown wherever an action needs the token and none is saved yet. The token is set only on the Settings page. */
export function NeedsToken({ token, loaded }: { token: string; loaded: boolean }) {
  if (!loaded || token !== "") return null;
  return (
    <p className="text-sm text-[var(--muted)]">
      Set the API token on the <Link href="/settings" className="underline">Settings</Link> page first.
    </p>
  );
}

/**
 * Asks for the exact name before a deletion. The token comes from the Settings page. The parent unmounts this
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
  const [token, , loaded] = useApiToken();
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
          <NeedsToken token={token} loaded={loaded} />
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

const PERSON_NAME = /^[a-z0-9_-]+$/;
const SESSION_NAME = /^session-\d{3}$/;

/** Asks for a new name. The name rules are checked here and again by the server. */
export function RenameDialog({
  title, current, kind = "person", pending = false, error, onClose, onConfirm,
}: {
  title: string;
  current: string;
  kind?: "person" | "session";
  pending?: boolean;
  error?: unknown;
  onClose: () => void;
  onConfirm: (token: string, newName: string) => void;
}) {
  const [token, , loaded] = useApiToken();
  const [name, setName] = useState("");
  const rule = kind === "session" ? SESSION_NAME : PERSON_NAME;
  const hint = kind === "session"
    ? "A recording name is session- followed by three digits, for example session-004."
    : "Use lowercase letters, numbers, hyphens or underscores.";
  const valid = rule.test(name) && name !== current;
  const ready = loaded && token !== "" && valid && !pending;
  return (
    <div role="dialog" aria-modal="true" aria-labelledby="rename-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onKeyDown={(e) => { if (e.key === "Escape") onClose(); }}>
      <div className="w-full max-w-md rounded-lg border border-[var(--border)] bg-white p-5 shadow-lg dark:bg-slate-900">
        <h2 id="rename-title" className="text-lg font-semibold">{title}</h2>
        <p className="mt-2 text-sm text-[var(--muted)]">
          Current name: <code className="font-mono">{current}</code>. {hint}
          {kind === "session" ? " A recording with processed data must have its data deleted first." : " A person with processed data must have its data deleted first."}
        </p>
        <div className="mt-4 flex flex-col gap-3">
          <NeedsToken token={token} loaded={loaded} />
          <label className="flex flex-col gap-1 text-sm">
            New name
            <input value={name} onChange={(e) => setName(e.target.value)} autoFocus
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono" />
          </label>
          {error ? <ErrorState {...errorText(error)} /> : null}
        </div>
        <div className="mt-5 flex justify-end gap-2 text-sm">
          <button type="button" onClick={onClose} className="rounded border border-[var(--border)] px-3 py-1.5">Cancel</button>
          <button type="button" disabled={!ready} onClick={() => onConfirm(token, name)}
            className="rounded border border-[var(--border)] px-3 py-1.5 disabled:opacity-40">
            {pending ? "Renaming…" : "Rename"}
          </button>
        </div>
      </div>
    </div>
  );
}
