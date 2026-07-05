"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";

const stages = [
  ["L1", "Decompose", "Atomic claims are extracted from the original text."],
  ["L2", "Retrieve", "Evidence is gathered and normalized across available sources."],
  ["L3", "Entail", "A cross-encoder scores support, contradiction, and unknown."],
  ["L4", "Graph", "Claim dependencies propagate through a probabilistic graph."],
  ["L5", "Calibrate", "Conformal intervals expose empirical uncertainty."],
] as const;

export function ArchitectureFlow() {
  const ref = useRef<HTMLElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-12% 0px" });

  return (
    <section ref={ref} className="sb-section bg-surface-base">
      <div className="sb-container">
        <motion.div
          className="max-w-4xl"
          initial={{ opacity: 0, y: 28 }}
          animate={isInView ? { opacity: 1, y: 0 } : undefined}
          transition={{ duration: 0.7 }}
        >
          <p className="sb-kicker">Architecture</p>
          <h2 className="mt-6">
            The v2 verification pipeline.
          </h2>
        </motion.div>

        <motion.div
          className="mt-14 grid gap-px overflow-hidden rounded-lg border border-[color:var(--border-subtle)] bg-[color:var(--border-subtle)] lg:grid-cols-5"
          initial={{ opacity: 0, y: 32 }}
          animate={isInView ? { opacity: 1, y: 0 } : undefined}
          transition={{ duration: 0.7, delay: 0.12 }}
        >
          {stages.map(([level, title, description], index) => (
            <motion.div
              key={level}
              className="bg-surface-base p-6 md:p-7"
              initial={{ opacity: 0, y: 18 }}
              animate={isInView ? { opacity: 1, y: 0 } : undefined}
              transition={{ duration: 0.55, delay: 0.22 + index * 0.06 }}
            >
              <p className="font-mono text-sm text-[color:var(--brand-accent)]">{level}</p>
              <h3 className="mt-7 font-mono text-3xl leading-tight">{title}</h3>
              <p className="mt-5 text-sm leading-6 text-brand-muted">{description}</p>
            </motion.div>
          ))}
        </motion.div>

        <div className="mt-8 flex flex-col gap-4 border-t border-[color:var(--border-subtle)] pt-8 md:flex-row md:items-center md:justify-between">
          <p className="max-w-2xl text-base text-brand-muted">
            Each layer is independently scored and cached. Degradation is surfaced per claim through
            explicit fallback metadata.
          </p>
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
    </section>
  );
}
