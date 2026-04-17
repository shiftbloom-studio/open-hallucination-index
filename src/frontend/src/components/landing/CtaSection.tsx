"use client";

import Link from "next/link";
import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { Button } from "@/components/ui/button";

function FloatingOrb({ className, delay = 0, style }: { className: string; delay?: number; style?: React.CSSProperties }) {
  return (
    <motion.div
      className={className}
      style={style}
      animate={{
        y: [-20, 20, -20],
        x: [-10, 10, -10],
        scale: [1, 1.08, 1],
      }}
      transition={{
        duration: 10,
        delay,
        repeat: Infinity,
        ease: "easeInOut",
      }}
    />
  );
}

export function CtaSection() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-10% 0px" });

  return (
    <section ref={ref} className="relative w-full">
      <div className="mx-auto max-w-7xl px-4 pb-20 md:pb-28">
        <motion.div
          className="relative overflow-hidden rounded-3xl border border-[color:var(--border-subtle)] p-10 md:p-16"
          style={{ background: "var(--surface-soft)" }}
          initial={{ opacity: 0, y: 50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <FloatingOrb
            className="absolute -top-20 -left-20 h-44 w-44 rounded-full"
            style={{ background: "var(--brand-indigo)", opacity: 0.12, filter: "blur(90px)" }}
            delay={0}
          />
          <FloatingOrb
            className="absolute -bottom-20 -right-20 h-56 w-56 rounded-full"
            style={{ background: "var(--brand-danger)", opacity: 0.08, filter: "blur(100px)" }}
            delay={2}
          />

          <div className="relative z-10 flex flex-col items-center text-center">
            <motion.h2
              className="font-display font-semibold tracking-tight text-brand-ink"
              style={{ fontSize: "clamp(2rem, 4.5vw, 3.5rem)", lineHeight: 1.05, letterSpacing: "-0.03em" }}
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.2 }}
            >
              Make hallucinations{" "}
              <motion.span
                className="display-accent"
                initial={{ opacity: 0, filter: "blur(6px)" }}
                animate={isInView ? { opacity: 1, filter: "blur(0px)" } : {}}
                transition={{ duration: 1, delay: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
              >
                measurable.
              </motion.span>
            </motion.h2>

            <motion.p
              className="mt-6 max-w-2xl text-base leading-relaxed text-brand-muted md:text-lg"
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.3 }}
            >
              Try it now. No account. No waitlist. Paste text, get a calibrated verdict with
              per-claim intervals and the full evidence graph.
            </motion.p>

            <motion.div
              className="mt-10 flex flex-col items-center gap-3 sm:flex-row"
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.4 }}
            >
              <Link href="/verify">
                <Button
                  size="lg"
                  className="h-12 rounded-full bg-brand-indigo px-7 text-[15px] font-semibold text-white shadow-sm transition-all hover:bg-[color:var(--brand-indigo-strong)] hover:shadow-md"
                >
                  Try /verify
                  <span className="ml-1" aria-hidden>
                    →
                  </span>
                </Button>
              </Link>

              <a
                href="https://github.com/shiftbloom-studio/open-hallucination-index"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button
                  size="lg"
                  variant="outline"
                  className="h-12 rounded-full border-[color:var(--border-default)] bg-surface-elevated/80 px-7 text-[15px] font-medium text-brand-ink backdrop-blur hover:bg-surface-elevated"
                >
                  API on GitHub
                </Button>
              </a>

              <Link href="/calibration">
                <Button
                  size="lg"
                  variant="ghost"
                  className="h-12 rounded-full px-7 text-[15px] font-medium text-brand-muted hover:bg-[color:var(--surface-soft)]/70 hover:text-brand-ink"
                >
                  Calibration report
                </Button>
              </Link>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
