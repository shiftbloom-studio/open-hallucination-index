"use client";

import Link from "next/link";
import { motion, type Variants, useReducedMotion } from "framer-motion";

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.12, delayChildren: 0.12 },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 28 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.72, ease: [0.22, 1, 0.36, 1] },
  },
};

export function HeroSection() {
  const prefersReducedMotion = useReducedMotion();
  const useMotion = !prefersReducedMotion;

  return (
    <section className="relative min-h-[920px] w-full overflow-hidden bg-surface-base pt-[236px] md:min-h-[980px]">
      <div className="sb-container">
        <motion.div
          className="max-w-5xl"
          variants={useMotion ? containerVariants : undefined}
          initial={useMotion ? "hidden" : false}
          animate={useMotion ? "visible" : undefined}
        >
          <motion.p variants={useMotion ? itemVariants : undefined} className="sb-kicker">
            Open Hallucination Index / Shiftbloom Studio
          </motion.p>

          <motion.h1 variants={useMotion ? itemVariants : undefined} className="sb-display mt-8">
            open hallucination{" "}
            <br />
            index.
          </motion.h1>

          <motion.p
            variants={useMotion ? itemVariants : undefined}
            className="mt-9 max-w-2xl text-[1.32rem] font-light leading-[1.625] text-brand-muted"
          >
            OHI decomposes AI-generated text into atomic claims and assigns each one a calibrated
            probability of being true, with explicit uncertainty intervals and an auditable claim graph.
          </motion.p>

          <motion.div
            variants={useMotion ? itemVariants : undefined}
            className="mt-10 flex flex-col items-start gap-4 sm:flex-row sm:items-center"
          >
            <Link href="/verify" className="modern-btn px-8">
              Test our work
            </Link>
            <Link href="/calibration" className="tertiary-btn">
              Read calibration
            </Link>
          </motion.div>
        </motion.div>

        <motion.div
          className="mt-24 grid max-w-4xl gap-8 border-t border-[color:var(--border-subtle)] pt-8 md:grid-cols-3"
          initial={useMotion ? { opacity: 0, y: 24 } : false}
          animate={useMotion ? { opacity: 1, y: 0 } : undefined}
          transition={{ duration: 0.72, delay: 0.55 }}
        >
          {[
            ["API", "/api/v2"],
            ["Score", "0.85 [0.78, 0.91]"],
            ["Output", "claim graph + evidence"],
          ].map(([label, value]) => (
            <div key={label}>
              <p className="sb-kicker">{label}</p>
              <p className="mt-2 font-mono text-sm text-brand-ink">{value}</p>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
