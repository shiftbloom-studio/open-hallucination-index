import { HeroSection } from "@/components/landing/HeroSection";
import { ProblemSection } from "@/components/landing/ProblemSection";
import { ArchitectureFlow } from "@/components/landing/ArchitectureFlow";
import { FeatureGrid } from "@/components/landing/FeatureGrid";
import { CtaSection } from "@/components/landing/CtaSection";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-slate-950 antialiased relative overflow-hidden">
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
