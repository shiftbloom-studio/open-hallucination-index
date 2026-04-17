import type { MetadataRoute } from "next";

// Required by `output: 'export'` — mark as a static asset.
export const dynamic = "force-static";

const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL ??
  (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : "http://localhost:3000");

const routes = [
  "",
  "/verify",
  "/calibration",
  "/status",
  "/about",
  "/accessibility",
  "/cookies",
  "/disclaimer",
  "/eula",
  "/agb",
  "/datenschutz",
  "/impressum",
];

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date();

  return routes.map((route) => ({
    url: `${siteUrl}${route}`,
    lastModified,
  }));
}
