import { redirect } from "next/navigation";

/** Diagnostics is part of the Face analysis page now. Older links are sent there. */
export default async function DiagnosticsRedirect({ params }: { params: Promise<{ person: string; session: string }> }) {
  const { person, session } = await params;
  redirect(`/people/${encodeURIComponent(person)}/sessions/${encodeURIComponent(session)}/facial`);
}
