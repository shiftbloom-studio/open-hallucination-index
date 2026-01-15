import type { Metadata } from "next";
import Script from "next/script";
import { Inter, Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { Toaster } from "sonner";
import { ParticlesBackground } from "@/components/ui/particles-background";
import { Navbar } from "@/components/layout/navbar";
import { CookieConsent } from "@/components/layout/cookie-consent";
import { ConsentAwareAnalytics } from "@/components/analytics/consent-aware-analytics";
import { Footer } from "@/components/layout/footer";


const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL ??
  (process.env.VERCEL_URL
    ? `https://${process.env.VERCEL_URL}`
    : "http://localhost:3000");
const siteName = "Open Hallucination Index";
const siteDescription =
  "Open-source project to measure and mitigate hallucinations in generative AI — practical tools, benchmarks and honest evaluations to improve model reliability.";

const structuredData = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Organization",
      "@id": `${siteUrl}/#organization`,
      name: siteName,
      url: siteUrl,
      logo: `${siteUrl}/logo_black.svg`,
    },
    {
      "@type": "WebSite",
      "@id": `${siteUrl}/#website`,
      url: siteUrl,
      name: siteName,
      description: siteDescription,
      publisher: {
        "@id": `${siteUrl}/#organization`,
      },
      inLanguage: "en",
    },
  ],
};

// Use a metadataBase if available from env, otherwise fall back to localhost for local dev.
export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  applicationName: siteName,
  title: {
    default: siteName,
    template: `%s | ${siteName}`,
  },
  description: siteDescription,
  keywords: [
    "AI safety",
    "hallucination",
    "factuality",
    "generative AI",
    "benchmark",
    "open-source",
  ],
  authors: [{ name: siteName, url: siteUrl }],
  creator: siteName,
  publisher: siteName,
  category: "Technology",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  // themeColor and colorScheme removed from metadata to comply with Next.js App Router
  // and avoid warnings; consider using `generateViewport` for viewport-related exports.
  icons: {
    icon: "/favicon.ico",
    shortcut: "/favicon.ico",
    apple: "/favicon.ico",
  },
  alternates: {
    canonical: siteUrl,
  },
  manifest: "/site.webmanifest",
  verification: {
    // keep empty keys if you don't have a verification token; these are placeholders
    google: process.env.NEXT_PUBLIC_GOOGLE_VERIFICATION ?? undefined,
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  openGraph: {
    title: siteName,
    description: siteDescription,
    url: siteUrl,
    siteName,
    type: "website",
    locale: "en_US",
    images: [
      // Primary large image (recommended 1200x630)
      {
        url: "/open-graph.jpg",
        width: 1200,
        height: 630,
        alt: "Open Hallucination Index — improving AI factuality",
        type: "image/jpeg",
      },
      // Medium variant for social previews
      {
        url: "/open-graph-800x418.jpg",
        width: 800,
        height: 418,
        alt: "Open Hallucination Index — improving AI factuality (800x418)",
        type: "image/jpeg",
      },
      // WebP versions for platforms that support modern formats
      {
        url: "/open-graph-1200x630.webp",
        width: 1200,
        height: 630,
        alt: "Open Hallucination Index — improving AI factuality (webp)",
        type: "image/webp",
      },
      {
        url: "/open-graph-800x418.webp",
        width: 800,
        height: 418,
        alt: "Open Hallucination Index — improving AI factuality (webp)",
        type: "image/webp",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: siteName,
    description: siteDescription,
    images: ["/open-graph-1200x630.webp", "/open-graph.jpg"],
    creator: "@openhallucindex",
    site: "@openhallucindex",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const jsonLd = JSON.stringify(structuredData).replace(/</g, "\\u003c");

  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${spaceGrotesk.variable} ${jetbrainsMono.variable}`}
    >
      <body className="font-sans antialiased">
        <Script
          id="structured-data"
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: jsonLd }}
        />
        <Providers>
          <ParticlesBackground />
          <Navbar />
          {children}
          <Footer />
          <CookieConsent />
          <ConsentAwareAnalytics />
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}
