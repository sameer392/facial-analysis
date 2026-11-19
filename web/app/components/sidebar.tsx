"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Clapperboard, Home, Users } from "lucide-react";

const NAV = [
  { href: "/", label: "Overview", icon: Home },
  { href: "/people", label: "People", icon: Users },
  { href: "/recordings", label: "Recordings", icon: Clapperboard },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <nav aria-label="Main" className="flex flex-col gap-1 p-4">
      <div className="mb-4 flex items-center gap-2 px-2 font-semibold">
        <Activity className="h-5 w-5" aria-hidden="true" />
        Facial Analysis
      </div>
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            className={`flex items-center gap-2 rounded px-3 py-2 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] ${
              active ? "bg-[var(--accent)]/15 font-medium" : "hover:bg-slate-500/10"
            }`}
          >
            <Icon className="h-4 w-4" aria-hidden="true" />
            {label}
          </Link>
        );
      })}
      <p className="mt-6 px-3 text-xs text-[var(--muted)]">Read-only. Phase P3.</p>
    </nav>
  );
}
