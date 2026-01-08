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
  
  // Security headers (compatible with Cloudflare)
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          { key: 'X-Accel-Buffering', value: 'no' },
          { key: 'X-DNS-Prefetch-Control', value: 'on' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          // CSP compatible with Cloudflare and inline scripts
          { 
            key: 'Content-Security-Policy', 
            value: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://challenges.cloudflare.com",
              "style-src 'self' 'unsafe-inline'",
              "img-src 'self' data: blob: https:",
              "font-src 'self' data:",
              "connect-src 'self' https: wss:",
              "frame-src 'self' https://challenges.cloudflare.com",
              "frame-ancestors 'none'",
              "form-action 'self'",
              "base-uri 'self'",
              "upgrade-insecure-requests"
            ].join('; ')
          },
          { key: 'Permissions-Policy', value: 'geolocation=(), microphone=(), camera=()' },
        ],
      },
    ];
  },
  
  // Prevent source map exposure in production
  productionBrowserSourceMaps: false,
  
  // Powered-by header removal
  poweredByHeader: false,
};

export default nextConfig;
