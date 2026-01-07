import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  experimental: {
    optimizeServerReact: true,
    optimizeCss: true,
    optimizePackageImports: [
      "lucide-react",
      "zod",
      "date-fns",
      "react-hook-form",
      "recharts",
      "@radix-ui/react-slot",
      "clsx",
      "tailwind-merge",
      "@react-three/fiber",
      "@react-three/drei"
    ],
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
  transpilePackages: ['three'],
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [{ key: 'X-Accel-Buffering', value: 'no' }],
      },
    ];
  },
};

export default nextConfig;
