"use client";

import Link from "next/link";
import { motion, useInView } from "framer-motion";
import { useRef } from "react";

export function CtaSection() {
  const ref = useRef<HTMLElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-12% 0px" });

  return (
    <section ref={ref} className="bg-surface-base pb-28 pt-12 md:pb-36">
      <div className="sb-container">
        <motion.div
          className="grid gap-10 border-t border-[color:var(--border-subtle)] pt-14 lg:grid-cols-[minmax(0,1fr)_360px] lg:items-end"
          initial={{ opacity: 0, y: 28 }}
          animate={isInView ? { opacity: 1, y: 0 } : undefined}
          transition={{ duration: 0.7 }}
        >
          <div>
            <p className="sb-kicker">Try it</p>
            <h2 className="mt-6 max-w-4xl">
              Make hallucinations measurable.
            </h2>
          </div>

          <div>
            <p className="text-base leading-7 text-brand-muted">
              Paste text, get calibrated per-claim verdicts, and inspect the evidence path.
            </p>
            <div className="mt-8 flex flex-col gap-4 sm:flex-row lg:flex-col">
              <Link href="/verify" className="modern-btn">
                Test our work
              </Link>
              <Link href="/status" className="tertiary-btn">
                System status
              </Link>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
