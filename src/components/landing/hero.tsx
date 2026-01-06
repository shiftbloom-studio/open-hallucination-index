"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Spotlight } from "@/components/ui/spotlight";
import { ButtonMovingBorder } from "@/components/ui/moving-border";
import dynamic from "next/dynamic";

const NeuralNetworkViz = dynamic(() => import("@/components/3d/neural-network"), { ssr: false });

export function LandingHero() {
  return (
    <div className="h-[40rem] w-full rounded-md flex md:items-center md:justify-center bg-black/[0.96] antialiased bg-grid-white/[0.02] relative overflow-hidden">
      <Spotlight className="-top-40 left-0 md:left-60 md:-top-20" fill="white" />
      <NeuralNetworkViz />
      <div className="p-4 max-w-7xl mx-auto relative z-10 w-full pt-20 md:pt-0">
        <h1 className="text-4xl md:text-7xl font-bold text-center bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400 bg-opacity-50">
          Enhancing AI Safety <br /> through Transparency.
        </h1>
        <p className="mt-4 font-normal text-base text-neutral-300 max-w-lg text-center mx-auto">
          An open-source initiative dedicated to measuring factual consistency and mitigating generation errors in modern Generative AI architectures.
        </p>
        <div className="flex gap-4 justify-center mt-8">
          <Link href="/auth/signup">
            <ButtonMovingBorder borderRadius="1.75rem" className="bg-slate-900 text-white border-slate-800">
               Get Started
            </ButtonMovingBorder>
          </Link>
          <Link href="/dashboard">
             <Button size="lg" variant="ghost" className="px-8 h-12 rounded-full border border-slate-700 text-neutral-300 hover:bg-slate-800/50 mt-2">
               View Dashboard
             </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
