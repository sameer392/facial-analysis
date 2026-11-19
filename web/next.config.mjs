/** Forward /api/* to the local read-only Python API (127.0.0.1 only). No other paths are proxied. */
const API_TARGET = process.env.FACIAL_API_URL ?? "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_TARGET}/api/:path*` }];
  },
};

export default nextConfig;
