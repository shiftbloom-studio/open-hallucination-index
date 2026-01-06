import Link from "next/link";
import { Button } from "@/components/ui/button";
import { HeroSection } from "@/components/landing/HeroSection";
import { ProblemSection } from "@/components/landing/ProblemSection";
import { ArchitectureFlow } from "@/components/landing/ArchitectureFlow";
import { FeatureGrid } from "@/components/landing/FeatureGrid";
import { CtaSection } from "@/components/landing/CtaSection";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-black/[0.96] antialiased relative overflow-hidden">
      <header className="border-b border-white/10 relative z-10 backdrop-blur-sm bg-black/40">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400">
            Open Hallucination Index
          </h1>
          <nav className="flex gap-4">
            <Link href="/auth/login">
              <Button variant="ghost" className="text-neutral-300 hover:text-white">Login</Button>
            </Link>
            <Link href="/auth/signup">
              <Button className="bg-slate-800 text-white border-slate-800">Sign Up</Button>
            </Link>
          </nav>
        </div>
      </header>

      <main className="flex-1 relative w-full">
        <HeroSection />
        <ProblemSection />
        <ArchitectureFlow />
        <FeatureGrid />
        <CtaSection />
      </main>
    </div>
  );
}
