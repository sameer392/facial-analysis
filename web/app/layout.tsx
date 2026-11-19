import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { Sidebar } from "./components/sidebar";

export const metadata: Metadata = {
  title: "Facial Analysis",
  description: "Read-only view of facial and body motion datasets (phases P0 to P3).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans antialiased">
        <Providers>
          <div className="grid min-h-screen grid-cols-1 md:grid-cols-[220px_1fr]">
            <aside className="border-b border-[var(--border)] md:border-b-0 md:border-r">
              <Sidebar />
            </aside>
            <main className="mx-auto w-full max-w-6xl p-6">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
