"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import Link from "next/link";
import { motion, type Variants, useReducedMotion } from "framer-motion";
import { Button } from "@/components/ui/button";

const KnowledgeGraphCanvas = dynamic(
  () => import("@/components/landing/_KnowledgeGraphCanvas"),
  { ssr: false },
);

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.2,
    },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 30, filter: "blur(10px)" },
  visible: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: 0.8 },
  },
};

export function HeroSection() {
  const prefersReducedMotion = useReducedMotion();
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const mediaQuery = window.matchMedia("(max-width: 767px)");
    const handleChange = (event: MediaQueryListEvent | MediaQueryList) => {
      setIsMobile(event.matches);
    };

    handleChange(mediaQuery);
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener("change", handleChange);
    } else {
      mediaQuery.addListener(handleChange);
    }

    return () => {
      if (mediaQuery.removeEventListener) {
        mediaQuery.removeEventListener("change", handleChange);
      } else {
        mediaQuery.removeListener(handleChange);
      }
    };
  }, []);

  const showHeroAnimations = !prefersReducedMotion && !isMobile;
  const showContentAnimations = !prefersReducedMotion;

  return (
    <section className="relative w-full overflow-hidden">
      {/* Lavender wash anchored at top center, fading to base surface */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 80% 60% at 50% -10%, var(--surface-soft) 0%, transparent 65%)",
        }}
      />
      {/* Ambient indigo glows */}
      {showHeroAnimations ? (
        <>
          <motion.div
            aria-hidden
            className="pointer-events-none absolute top-32 right-10 h-64 w-64 rounded-full"
            style={{ background: "var(--brand-indigo)", opacity: 0.14, filter: "blur(110px)" }}
            animate={{ scale: [1, 1.15, 1], opacity: [0.12, 0.2, 0.12] }}
            transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
          />
          <motion.div
            aria-hidden
            className="pointer-events-none absolute bottom-10 left-10 h-80 w-80 rounded-full"
            style={{ background: "var(--brand-danger)", opacity: 0.07, filter: "blur(130px)" }}
            animate={{ scale: [1.1, 1, 1.1], opacity: [0.06, 0.1, 0.06] }}
            transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
          />
        </>
      ) : (
        <>
          <div
            aria-hidden
            className="pointer-events-none absolute top-32 right-10 h-64 w-64 rounded-full"
            style={{ background: "var(--brand-indigo)", opacity: 0.1, filter: "blur(110px)" }}
          />
          <div
            aria-hidden
            className="pointer-events-none absolute bottom-10 left-10 h-80 w-80 rounded-full"
            style={{ background: "var(--brand-danger)", opacity: 0.06, filter: "blur(130px)" }}
          />
        </>
      )}
      {showHeroAnimations && <KnowledgeGraphCanvas />}

      <div className="relative z-10 mx-auto flex max-w-5xl flex-col items-center px-4 pt-24 pb-20 md:pt-32 md:pb-28">
        <motion.div
          variants={showContentAnimations ? containerVariants : undefined}
          initial={showContentAnimations ? "hidden" : false}
          animate={showContentAnimations ? "visible" : undefined}
          className="flex flex-col items-center w-full"
        >
          <motion.div variants={showContentAnimations ? itemVariants : undefined}>
            <span className="inline-flex items-center gap-2 rounded-full border border-[color:var(--border-subtle)] bg-surface-elevated/75 px-3 py-1 text-xs font-medium text-brand-muted backdrop-blur">
              <motion.span
                className="h-1.5 w-1.5 rounded-full"
                style={{ background: "var(--brand-success)" }}
                animate={showHeroAnimations ? { scale: [1, 1.35, 1], opacity: [1, 0.7, 1] } : undefined}
                transition={showHeroAnimations ? { duration: 2, repeat: Infinity } : undefined}
              />
              Open Hallucination Index
              <span className="h-1 w-1 rounded-full bg-brand-subtle" />
              Open Source
            </span>
          </motion.div>

          <motion.h1
            variants={showContentAnimations ? itemVariants : undefined}
            className="mt-8 text-center font-display font-semibold tracking-tight text-brand-ink"
            style={{ fontSize: "clamp(2.75rem, 6.5vw, 5.5rem)", lineHeight: 1.02, letterSpacing: "-0.035em" }}
          >
            How much should you{" "}
            <span className="display-accent">trust</span>
            <br className="hidden md:inline" />
            {" "}that answer?
          </motion.h1>

          <motion.p
            variants={showContentAnimations ? itemVariants : undefined}
            className="mt-6 max-w-2xl text-center text-base leading-relaxed text-brand-muted md:text-lg"
          >
            OHI decomposes AI-generated text into atomic claims and assigns each one a{" "}
            <span className="font-medium text-brand-ink">calibrated probability</span> of being
            true — with an explicit uncertainty interval. No black-box confidence. No silent
            failures.
          </motion.p>

          <motion.div
            variants={showContentAnimations ? itemVariants : undefined}
            className="mt-10 flex flex-col items-center gap-3 sm:flex-row"
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
            <Link href="/calibration">
              <Button
                size="lg"
                variant="outline"
                className="h-12 rounded-full border-[color:var(--border-default)] bg-surface-elevated/70 px-7 text-[15px] font-medium text-brand-ink backdrop-blur hover:bg-surface-elevated"
              >
                Read the calibration report
              </Button>
            </Link>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
