"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type FeatureAccent = "indigo" | "danger" | "success";

const features: {
  title: string;
  description: string;
  icon: string;
  accent: FeatureAccent;
  colSpan: string;
}[] = [
  {
    title: "Calibrated probabilities",
    description:
      "Not black-box confidence. Per-domain split conformal prediction gives you intervals with empirical coverage you can audit — 0.85 [0.78, 0.91] at 90% target means the guarantee, not the vibe.",
    icon: "🎯",
    accent: "indigo",
    colSpan: "lg:col-span-3",
  },
  {
    title: "Probabilistic Claim Graph",
    description:
      "Entailment and contradiction edges between claims propagate evidence through a loopy graph (TRW-BP). A refuted claim drags its dependencies. A contradiction pair can't both be 0.9.",
    icon: "🕸️",
    accent: "danger",
    colSpan: "lg:col-span-3",
  },
  {
    title: "Open, auditable, rest-respecting",
    description:
      "Daily calibration report is public. Methodology lives in a single open spec. When the PC is off, we say so — not 'temporarily unavailable'.",
    icon: "🌅",
    accent: "success",
    colSpan: "lg:col-span-6",
  },
];

const accentMap: Record<FeatureAccent, { ring: string; iconBg: string; iconFg: string }> = {
  indigo: { ring: "rgba(99,102,241,0.35)", iconBg: "rgba(99,102,241,0.12)", iconFg: "var(--brand-indigo-strong)" },
  danger: { ring: "rgba(230,57,70,0.35)", iconBg: "rgba(230,57,70,0.1)", iconFg: "var(--brand-danger)" },
  success: { ring: "rgba(5,150,105,0.35)", iconBg: "rgba(5,150,105,0.1)", iconFg: "var(--brand-success)" },
};

export function FeatureGrid() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-10% 0px" });

  return (
    <section ref={ref} className="relative w-full">
      <div className="mx-auto max-w-7xl px-4 py-16 md:py-24">
        <motion.div
          className="mx-auto max-w-3xl text-center"
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }}
        >
          <span className="label-mono" style={{ color: "var(--brand-indigo-strong)" }}>
            Features
          </span>
          <h2
            className="mt-3 font-display font-semibold tracking-tight text-brand-ink"
            style={{ fontSize: "clamp(2rem, 4.5vw, 3.5rem)", lineHeight: 1.05, letterSpacing: "-0.03em" }}
          >
            Not another <span className="display-accent">confidence</span> score.
          </h2>
          <p className="mt-4 text-base leading-relaxed text-brand-muted md:text-lg">
            Calibrated probabilities, a probabilistic claim graph, and a public audit trail.
          </p>
        </motion.div>

        <div className="mt-10 grid gap-4 lg:grid-cols-6">
          {features.map((feature, index) => {
            const a = accentMap[feature.accent];
            return (
              <motion.div
                key={feature.title}
                className={feature.colSpan}
                initial={{ opacity: 0, y: 40, scale: 0.97 }}
                animate={isInView ? { opacity: 1, y: 0, scale: 1 } : {}}
                transition={{ duration: 0.6, delay: 0.15 + index * 0.1 }}
              >
                <motion.div
                  whileHover={{ y: -4, transition: { duration: 0.2 } }}
                  className="h-full"
                >
                  <Card
                    className={cn(
                      "h-full relative overflow-hidden border-[color:var(--border-subtle)] bg-surface-elevated shadow-sm transition-all duration-300 group cursor-pointer",
                    )}
                    style={{
                      // subtle lavender wash on hover via the ring border color
                    }}
                  >
                    <div
                      className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100"
                      style={{
                        background: `radial-gradient(120% 60% at 0% 0%, ${a.ring} 0%, transparent 55%)`,
                      }}
                    />

                    <CardHeader className="relative z-10">
                      <div className="flex items-center gap-3">
                        <motion.span
                          className="inline-flex h-10 w-10 items-center justify-center rounded-lg text-lg"
                          style={{ background: a.iconBg, color: a.iconFg }}
                          animate={isInView ? { rotate: [0, 8, -6, 0] } : {}}
                          transition={{ duration: 0.5, delay: 0.5 + index * 0.1 }}
                        >
                          {feature.icon}
                        </motion.span>
                        <CardTitle className="font-heading text-lg font-semibold text-brand-ink md:text-xl">
                          {feature.title}
                        </CardTitle>
                      </div>
                    </CardHeader>
                    <CardContent className="relative z-10 text-sm leading-relaxed text-brand-muted md:text-base">
                      {feature.description}
                    </CardContent>
                  </Card>
                </motion.div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
