import type { ReactNode } from "react";
import { AlertTriangle, Inbox, Loader2 } from "lucide-react";

export function Card({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-[var(--border)] bg-white/60 p-5 shadow-sm dark:bg-white/[0.03]">
      {title ? <h2 className="mb-3 text-sm font-semibold text-[var(--muted)]">{title}</h2> : null}
      {children}
    </section>
  );
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "good" | "warn" }) {
  const tones = {
    neutral: "bg-slate-200 text-slate-800 dark:bg-slate-700 dark:text-slate-100",
    good: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-100",
    warn: "bg-amber-100 text-amber-900 dark:bg-amber-900 dark:text-amber-100",
  };
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${tones[tone]}`}>{children}</span>;
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-2 py-6 text-sm text-[var(--muted)]">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
      {label}
    </div>
  );
}

export function Empty({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="flex flex-col items-center gap-2 py-8 text-center text-sm text-[var(--muted)]">
      <Inbox className="h-6 w-6" aria-hidden="true" />
      <p className="font-medium text-[var(--foreground)]">{title}</p>
      {detail ? <p>{detail}</p> : null}
    </div>
  );
}

export function ErrorState({ message, code }: { message: string; code?: string }) {
  return (
    <div role="alert" className="flex items-start gap-2 rounded border border-red-300 bg-red-50 p-3 text-sm text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
      <div>
        <p className="font-medium">Could not load data</p>
        <p>{message}</p>
        {code ? <p className="text-xs opacity-70">code: {code}</p> : null}
      </div>
    </div>
  );
}
