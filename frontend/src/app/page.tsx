import { HeroSection } from "@/components/landing/HeroSection";
import { ProblemSection } from "@/components/landing/ProblemSection";
import { ArchitectureFlow } from "@/components/landing/ArchitectureFlow";
import { FeatureGrid } from "@/components/landing/FeatureGrid";
import { CtaSection } from "@/components/landing/CtaSection";
import { ParticlesBackground } from "@/components/ui/particles-background";
import { SmoothScroll } from "@/components/ui/smooth-scroll";

export default function Home() {
  return (
    <SmoothScroll>
      <div className="min-h-screen flex flex-col bg-slate-950 antialiased relative overflow-hidden">
        <ParticlesBackground />
        <main className="flex-1 relative w-full">
          <HeroSection />
          <ProblemSection />
          <ArchitectureFlow />
          <FeatureGrid />
          <CtaSection />
        </main>
      </div>
    </SmoothScroll>
  );
}
