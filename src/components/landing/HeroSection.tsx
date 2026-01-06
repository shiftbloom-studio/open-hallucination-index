"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ButtonMovingBorder } from "@/components/ui/moving-border";
import { Spotlight } from "@/components/ui/spotlight";

const KnowledgeGraphCanvas = dynamic(
  () => import("@/components/landing/_KnowledgeGraphCanvas"),
  { ssr: false }
);

export function HeroSection() {
  return (
    <section className="relative w-full overflow-hidden">
      <div className="relative h-[44rem] w-full bg-black/[0.96] antialiased bg-grid-white/[0.02]">
        <Spotlight className="-top-40 left-0 md:left-60 md:-top-20" fill="white" />
        <KnowledgeGraphCanvas />

        <div className="relative z-10 mx-auto flex h-full max-w-7xl flex-col items-center justify-center px-4 pt-24 md:pt-0">
          <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-neutral-200 backdrop-blur">
            Open Hallucination Index
            <span className="h-1 w-1 rounded-full bg-white/40" />
            Open Source
          </p>

          <h1 className="text-center text-4xl font-bold tracking-tight text-transparent md:text-7xl bg-clip-text bg-gradient-to-b from-neutral-50 to-neutral-400">
            The Trust Layer for Artificial Intelligence.
          </h1>

          <p className="mt-6 max-w-2xl text-center text-base text-neutral-300 md:text-lg">
            Decompose outputs into verifiable claims, route them through a verification oracle, and surface a trust score you can act on.
          </p>

          <div className="mt-10 flex flex-col items-center gap-4 sm:flex-row">
            <Link href="/auth/signup">
              <ButtonMovingBorder
                borderRadius="1.75rem"
                className="bg-slate-900 text-white border-slate-800"
              >
                Get Started
              </ButtonMovingBorder>
            </Link>
            <Link href="/pricing">
              <Button
                size="lg"
                variant="ghost"
                className="h-12 rounded-full border border-white/10 bg-white/5 px-8 text-neutral-200 hover:bg-white/10"
              >
                See Pricing
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
