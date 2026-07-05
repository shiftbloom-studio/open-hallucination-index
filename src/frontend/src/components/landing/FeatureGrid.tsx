"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";

const features = [
  {
    title: "Calibrated probabilities",
    description:
      "Per-domain split conformal prediction gives each claim a probability and interval that can be audited against the public calibration report.",
  },
  {
    title: "Probabilistic claim graph",
    description:
      "Entailment and contradiction edges move evidence across related claims instead of treating every sentence as isolated.",
  },
  {
    title: "Open, rest-respecting API",
    description:
      "The API reports when a subsystem falls back or degrades, and returns evidence paths rather than hiding uncertainty behind status text.",
  },
] as const;

export function FeatureGrid() {
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
          <p className="sb-kicker">What changes</p>
          <h2 className="mt-6">
            Not another confidence score.
          </h2>
        </motion.div>

        <div className="mt-14 grid gap-10 lg:grid-cols-3">
          {features.map((feature, index) => (
            <motion.article
              key={feature.title}
              className="border-t border-[color:var(--border-subtle)] pt-7"
              initial={{ opacity: 0, y: 22 }}
              animate={isInView ? { opacity: 1, y: 0 } : undefined}
              transition={{ duration: 0.55, delay: 0.15 + index * 0.08 }}
            >
              <span className="font-mono text-sm text-[color:var(--brand-accent)]">
                0{index + 1}
              </span>
              <h3 className="mt-8 text-3xl leading-tight">{feature.title}</h3>
              <p className="mt-5 text-base leading-7 text-brand-muted">{feature.description}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}
