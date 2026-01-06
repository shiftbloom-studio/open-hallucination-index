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
import Link from "next/link";

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
             <footer className="border-t border-white/10 py-10 relative z-10 bg-black">
               <div className="container mx-auto px-4 text-center text-neutral-500 space-y-4">
                 <nav className="flex flex-wrap justify-center gap-x-6 gap-y-2 text-sm">
                   <Link className="hover:text-neutral-200" href="/impressum">Legal Notice</Link>
                   <Link className="hover:text-neutral-200" href="/agb">Terms of Service</Link>
                   <Link className="hover:text-neutral-200" href="/datenschutz">Privacy Policy</Link>
                   <Link className="hover:text-neutral-200" href="/cookies">Cookie Policy</Link>
                   <Link className="hover:text-neutral-200" href="/eula">EULA</Link>
                   <Link className="hover:text-neutral-200" href="/disclaimer">Disclaimer</Link>
                   <Link className="hover:text-neutral-200" href="/accessibility">Accessibility</Link>
                 </nav>
                 <p className="text-xs">&copy; {new Date().getFullYear()} Open Hallucination Index.</p>
               </div>
             </footer>
             <CookieConsent />
             <ConsentAwareAnalytics />
             <Toaster />
          </SmoothScroll>
        </Providers>
      </body>
    </html>
  );
}
