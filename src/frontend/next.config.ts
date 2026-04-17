import type { NextConfig } from 'next';

// Static export. We ship the frontend to Vercel via `output: 'export'`;
// the API is cross-origin on api.ohi.shiftbloom.studio (Cloudflare-fronted
// Lambda). Browser CSP / security headers move to the deploy platform
// (Vercel / Cloudflare) — the `async headers()` hook does not apply to a
// static export.

const nextConfig: NextConfig = {
  output: 'export',
  // Static export requires self-optimizing images disabled (no server loader).
  images: { unoptimized: true },
  trailingSlash: false,
  experimental: {
    optimizeCss: true,
    optimizePackageImports: [
      "lucide-react",
      "zod",
      "@radix-ui/react-slot",
      "clsx",
      "tailwind-merge",
      "@react-three/fiber",
      "@react-three/drei",
    ],
  },
  transpilePackages: ['three'],
  productionBrowserSourceMaps: false,
  poweredByHeader: false,
};

export default nextConfig;
