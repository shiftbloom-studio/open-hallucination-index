import type { MetadataRoute } from "next";

const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL ??
  (process.env.NODE_ENV === "development"
    ? "http://localhost:3000"
    : "https://open-hallucination-index.org");

const routes = [
  "",
  "/about",
  "/pricing",
  "/agb",
  "/accessibility",
  "/cookies",
  "/datenschutz",
  "/disclaimer",
  "/eula",
  "/impressum",
];

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date();

  return routes.map((route) => ({
    url: `${siteUrl}${route}`,
    lastModified,
  }));
}
