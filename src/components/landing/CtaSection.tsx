"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ButtonMovingBorder } from "@/components/ui/moving-border";

export function CtaSection() {
  return (
    <section className="relative w-full">
      <div className="mx-auto max-w-7xl px-4 pb-24 md:pb-32">
        <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-black/50 p-10 backdrop-blur md:p-16">
          <div className="pointer-events-none absolute inset-0 opacity-70">
            <div className="absolute -inset-24 animate-pulse bg-gradient-to-r from-white/10 via-transparent to-white/10 blur-3xl" />
          </div>

          <div className="relative z-10 flex flex-col items-center text-center">
            <h2 className="text-3xl font-bold tracking-tight text-neutral-50 md:text-5xl">
              Make hallucinations measurable.
            </h2>
            <p className="mt-6 max-w-2xl text-base leading-relaxed text-neutral-300 md:text-lg">
              Start verifying today—run it locally, integrate it into your pipeline, and ship AI you can trust.
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

              <Link
                href="https://github.com/shiftbloom-studio/open-hallucination-index"
                target="_blank"
                rel="noreferrer"
              >
                <Button
                  size="lg"
                  variant="ghost"
                  className="h-12 rounded-full border border-white/10 bg-white/5 px-8 text-neutral-200 hover:bg-white/10"
                >
                  View on GitHub
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
