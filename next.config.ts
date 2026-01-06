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
};

export default nextConfig;
