import type { Metadata } from "next";
import { Inter, Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { Toaster } from "sonner";
import { SmoothScroll } from "@/components/ui/smooth-scroll";
import { ParticlesBackground } from "@/components/ui/particles-background";
import { Navbar } from "@/components/layout/navbar";
import { CookieConsent } from "@/components/layout/cookie-consent";
import { ConsentAwareAnalytics } from "@/components/analytics/consent-aware-analytics";
import { Footer } from "@/components/layout/footer";
import { Analytics } from "@vercel/analytics/react";

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

export const metadata: Metadata = {
  title: "Open Hallucination Index",
  description: "An open-source initiative dedicated to enhancing AI safety by providing a robust toolkit for measuring factual consistency and mitigating generation errors in modern Generative AI architectures.",
  icons: {
    icon: "/favicon.ico",
    shortcut: "/favicon.ico",
    apple: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className={`${inter.variable} ${spaceGrotesk.variable} ${jetbrainsMono.variable}`}>
      <body className="font-sans antialiased">
        <Providers>
          <SmoothScroll>
             <ParticlesBackground />
             <Navbar />
             {children}
             <Footer />
             <CookieConsent />
             <ConsentAwareAnalytics />
             <Toaster />
          </SmoothScroll>
        </Providers>
        <Analytics />
      </body>
    </html>
  );
}
