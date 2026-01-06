import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { Toaster } from "sonner";
import { SmoothScroll } from "@/components/ui/smooth-scroll";
import { ParticlesBackground } from "@/components/ui/particles-background";

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
             <Toaster />
          </SmoothScroll>
        </Providers>
      </body>
    </html>
  );
}
