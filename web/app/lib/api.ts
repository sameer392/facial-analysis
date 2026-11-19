/** Typed client for the read-only Python API. Requests go to /api, which Next.js proxies to 127.0.0.1. */

export type Health = {
  status: string;
  api_version: string;
  schema_version: string;
  data_root: string;
  read_only: boolean;
};

export type SessionRow = {
  session_id: string;
  has_video: boolean;
  layout: "v1" | "v2" | "incomplete_v2" | "none";
};

export class ApiFailure extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`/api${path}`, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    let code = "http_error";
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { code?: string; message?: string };
      code = body.code ?? code;
      message = body.message ?? message;
    } catch {
      // not JSON: keep the generic message
    }
    throw new ApiFailure(response.status, code, message);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => get<Health>("/health"),
  persons: () => get<string[]>("/persons"),
  sessions: (person: string) => get<SessionRow[]>(`/persons/${encodeURIComponent(person)}/sessions`),
};
