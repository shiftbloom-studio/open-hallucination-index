"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";

const rows = [
  {
    prompt: "The Eiffel Tower was built in 1920.",
    result: "0.04 probability true",
    detail: "Refuted by primary and encyclopedia evidence; date corrected to 1889.",
  },
  {
    prompt: "Water boils at 100°C at sea level.",
    result: "0.93 probability true",
    detail: "Supported by multiple sources; interval remains tight.",
  },
  {
    prompt: "Shakespeare wrote 47 plays.",
    result: "0.18 probability true",
    detail: "Evidence disagrees with the exact number; uncertainty stays visible.",
  },
];

export function ProblemSection() {
  const ref = useRef<HTMLElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-12% 0px" });

  return (
    <section ref={ref} className="sb-section bg-surface-base">
      <div className="sb-container">
        <div className="grid gap-16 lg:grid-cols-[minmax(0,0.9fr)_minmax(420px,1fr)] lg:items-start">
          <motion.div
            initial={{ opacity: 0, y: 28 }}
            animate={isInView ? { opacity: 1, y: 0 } : undefined}
            transition={{ duration: 0.7 }}
          >
            <p className="sb-kicker">Why it exists</p>
            <h2 className="mt-6 max-w-4xl">
              Confidence is not calibration.
            </h2>
            <p className="mt-8 max-w-xl text-[1.32rem] font-light leading-[1.625] text-brand-muted">
              Most AI products collapse uncertainty into a single reassuring answer. OHI keeps the
              claim, evidence, probability, and interval separate so failures stay inspectable.
            </p>
          </motion.div>

          <motion.div
            className="sb-panel p-4 md:p-6"
            initial={{ opacity: 0, y: 32 }}
            animate={isInView ? { opacity: 1, y: 0 } : undefined}
            transition={{ duration: 0.7, delay: 0.15 }}
          >
            <div className="flex items-center justify-between border-b border-[color:var(--border-subtle)] pb-4">
              <p className="sb-kicker">Claim Audit</p>
              <span className="rounded-full bg-[color:var(--brand-secondary)] px-3 py-1 font-mono text-xs text-[color:var(--brand-accent)]">
                live schema
              </span>
            </div>

            <div className="divide-y divide-[color:var(--border-subtle)]">
              {rows.map((row, index) => (
                <motion.div
                  key={row.prompt}
                  className="grid gap-4 py-5 md:grid-cols-[1fr_180px]"
                  initial={{ opacity: 0, x: 18 }}
                  animate={isInView ? { opacity: 1, x: 0 } : undefined}
                  transition={{ duration: 0.55, delay: 0.25 + index * 0.08 }}
                >
                  <div>
                    <p className="text-base font-medium text-brand-ink">{row.prompt}</p>
                    <p className="mt-2 text-sm leading-6 text-brand-muted">{row.detail}</p>
                  </div>
                  <p className="font-mono text-sm text-brand-ink md:text-right">{row.result}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
