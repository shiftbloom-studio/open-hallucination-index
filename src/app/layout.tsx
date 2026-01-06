import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { Toaster } from "sonner";
import { SmoothScroll } from "@/components/ui/smooth-scroll";
import { ParticlesBackground } from "@/components/ui/particles-background";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Open Hallucination Index",
  description: "An open-source initiative dedicated to enhancing AI safety by providing a robust toolkit for measuring factual consistency and mitigating generation errors in modern Generative AI architectures.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased">
        <Providers>
          <SmoothScroll>
             <ParticlesBackground />
             {children}
             <footer className="border-t border-white/10 py-10 relative z-10 bg-black">
               <div className="container mx-auto px-4 text-center text-neutral-500 space-y-4">
                 <nav className="flex flex-wrap justify-center gap-x-6 gap-y-2 text-sm">
                   <Link className="hover:text-neutral-200" href="/impressum">Impressum</Link>
                   <Link className="hover:text-neutral-200" href="/agb">AGB</Link>
                   <Link className="hover:text-neutral-200" href="/datenschutz">Datenschutz</Link>
                 </nav>
                 <p className="text-xs">&copy; {new Date().getFullYear()} Open Hallucination Index.</p>
               </div>
             </footer>
             <Toaster />
          </SmoothScroll>
        </Providers>
      </body>
    </html>
  );
}
