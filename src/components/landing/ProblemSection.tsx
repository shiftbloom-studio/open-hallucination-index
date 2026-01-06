"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";

export function ProblemSection() {
  const ref = useRef<HTMLDivElement | null>(null);
  const isInView = useInView(ref, { once: true, margin: "-15% 0px" });

  return (
    <section className="relative w-full">
      <div className="mx-auto max-w-7xl px-4 py-20 md:py-28">
        <div
          ref={ref}
          className="relative overflow-hidden rounded-3xl border border-white/10 bg-black/40 px-6 py-14 backdrop-blur md:px-14"
        >
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-white/10 via-transparent to-transparent" />

          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 12 }}
            transition={{ duration: 0.7 }}
            className="text-sm font-medium tracking-wide text-neutral-300"
          >
            The Problem
          </motion.p>

          <motion.h2
            initial={{ opacity: 0, y: 18 }}
            animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 18 }}
            transition={{ duration: 0.85, delay: 0.05 }}
            className="mt-4 text-3xl font-bold tracking-tight text-neutral-50 md:text-5xl"
          >
            LLMs Hallucinate. We Verify.
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 18 }}
            animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 18 }}
            transition={{ duration: 0.85, delay: 0.12 }}
            className="mt-6 max-w-3xl text-base leading-relaxed text-neutral-300 md:text-lg"
          >
            Modern models can sound certain while being wrong. OHI turns that confidence into measurable trust by extracting concrete claims and verifying them against grounded evidence.
          </motion.p>
        </div>
      </div>
    </section>
  );
}
