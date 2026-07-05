"use client";

import Link from "next/link";
import { motion, useInView } from "framer-motion";
import { useRef } from "react";

const principles = [
  {
    title: "Calibration over confidence",
    description:
      "Every score is designed to be compared against observed coverage, not treated as a decorative certainty badge.",
  },
  {
    title: "Evidence stays inspectable",
    description:
      "Claims, sources, intervals, fallbacks, and graph relationships are kept separate so reviewers can see what happened.",
  },
  {
    title: "Open systems age better",
    description:
      "The methodology, API contracts, and calibration reports are public so independent teams can challenge the work.",
  },
] as const;

function Reveal({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-12% 0px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 28 }}
      animate={isInView ? { opacity: 1, y: 0 } : undefined}
      transition={{ duration: 0.7 }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

export default function AboutPage() {
  return (
    <main className="bg-surface-base">
      <section className="min-h-[840px] pt-[236px]">
        <div className="sb-container">
          <Reveal className="max-w-5xl">
            <p className="sb-kicker">Studio</p>
            <h1 className="mt-8">
              Verification as open infrastructure.
            </h1>
            <p className="mt-9 max-w-2xl text-[1.32rem] font-light leading-[1.625] text-brand-muted">
              Open Hallucination Index is a Shiftbloom Studio project for turning factuality checks
              into something measurable, inspectable, and useful to builders.
            </p>
          </Reveal>
        </div>
      </section>

      <section className="sb-section">
        <div className="sb-container">
          <Reveal className="grid gap-16 lg:grid-cols-[minmax(0,0.85fr)_minmax(420px,1fr)]">
            <div>
              <p className="sb-kicker">Mission</p>
              <h2 className="mt-6">Truth is not a vibe.</h2>
            </div>
            <div className="space-y-7 pt-2">
              <p className="text-[1.32rem] font-light leading-[1.625] text-brand-muted">
                AI systems increasingly produce answers that sound finished before they are checked.
                OHI gives teams a sharper interface: decompose the answer, score the claims, expose
                the uncertainty, and keep the evidence path visible.
              </p>
              <p className="text-base leading-7 text-brand-muted">
                The project is intentionally plain about failure states. If retrieval, graph scoring,
                calibration, or model inference degrades, the API reports that state instead of hiding it.
              </p>
            </div>
          </Reveal>
        </div>
      </section>

      <section className="sb-section">
        <div className="sb-container">
          <Reveal>
            <p className="sb-kicker">Principles</p>
            <h2 className="mt-6 max-w-4xl">Built for review.</h2>
            <div className="mt-14 grid gap-10 lg:grid-cols-3">
              {principles.map((principle, index) => (
                <article key={principle.title} className="border-t border-[color:var(--border-subtle)] pt-7">
                  <span className="font-mono text-sm text-[color:var(--brand-accent)]">0{index + 1}</span>
                  <h3 className="mt-8 text-3xl leading-tight">{principle.title}</h3>
                  <p className="mt-5 text-base leading-7 text-brand-muted">{principle.description}</p>
                </article>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      <section className="pb-28 pt-12 md:pb-36">
        <div className="sb-container">
          <Reveal className="grid gap-10 border-t border-[color:var(--border-subtle)] pt-14 lg:grid-cols-[minmax(0,1fr)_360px] lg:items-end">
            <div>
              <p className="sb-kicker">Use it</p>
              <h2 className="mt-6 max-w-4xl">Bring calibration into your workflow.</h2>
            </div>
            <div>
              <p className="text-base leading-7 text-brand-muted">
                Start in the browser, then wire the same API into reviews, evaluations, or internal tools.
              </p>
              <div className="mt-8 flex flex-col gap-4 sm:flex-row lg:flex-col">
                <Link href="/verify" className="modern-btn">
                  Test our work
                </Link>
                <a
                  href="https://github.com/shiftbloom-studio/open-hallucination-index"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="tertiary-btn"
                >
                  View source
                </a>
              </div>
            </div>
          </Reveal>
        </div>
      </section>
    </main>
  );
}
