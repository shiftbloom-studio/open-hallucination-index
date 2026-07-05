import type { NextConfig } from 'next';

// Static export. Cloudflare Workers serves these assets and handles the API
// under the same origin, so platform headers live in the Worker deployment.

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
