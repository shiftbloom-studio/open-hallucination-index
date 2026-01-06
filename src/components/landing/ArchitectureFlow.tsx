"use client";

import { motion, useInView } from "framer-motion";
import { useMemo, useRef } from "react";

const steps = [
  {
    title: "Input Text",
    description: "User or system prompt enters the pipeline.",
  },
  {
    title: "Claim Decomposition",
    description: "Atomize output into discrete, verifiable statements.",
  },
  {
    title: "Verification Oracle",
    description: "Graph + vector search retrieve evidence and check claims.",
  },
  {
    title: "Trust Score",
    description: "Surface confidence you can route, log, and enforce.",
  },
] as const;

function wrapSvgText(text: string, maxCharsPerLine: number) {
  const words = text.trim().split(/\s+/);
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= maxCharsPerLine) {
      current = candidate;
      continue;
    }

    if (current) lines.push(current);
    current = word;
  }

  if (current) lines.push(current);
  return lines;
}

export function ArchitectureFlow() {
  const ref = useRef<HTMLDivElement | null>(null);
  const isInView = useInView(ref, { once: true, margin: "-15% 0px" });

  const dash = useMemo(() => 220, []);

  return (
    <section className="relative w-full">
      <div className="mx-auto max-w-7xl px-4 py-20 md:py-28">
        <div ref={ref} className="grid gap-10 lg:grid-cols-2 lg:items-center">
          <div>
            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 10 }}
              transition={{ duration: 0.6 }}
              className="text-sm font-medium tracking-wide text-neutral-300"
            >
              Architecture
            </motion.p>
            <motion.h2
              initial={{ opacity: 0, y: 14 }}
              animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 14 }}
              transition={{ duration: 0.75, delay: 0.05 }}
              className="mt-4 text-3xl font-bold tracking-tight text-neutral-50 md:text-5xl"
            >
              Verification, step by step.
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 14 }}
              animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 14 }}
              transition={{ duration: 0.75, delay: 0.12 }}
              className="mt-6 max-w-xl text-base leading-relaxed text-neutral-300 md:text-lg"
            >
              OHI treats generations like software: break them into units, check them against sources, and expose a score that can gate downstream actions.
            </motion.p>
          </div>

          <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-black/40 p-6 backdrop-blur md:p-10">
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-white/10 via-transparent to-transparent" />

            <motion.svg
              viewBox="0 0 820 340"
              className="relative z-10 h-auto w-full text-neutral-100"
              initial={false}
            >
              <motion.path
                d="M 130 170 C 240 50, 330 50, 440 170 S 640 290, 710 170"
                fill="none"
                stroke="currentColor"
                strokeOpacity="0.35"
                strokeWidth="2"
                strokeLinecap="round"
                strokeDasharray={dash}
                strokeDashoffset={isInView ? 0 : dash}
                transition={{ duration: 1.15, ease: "easeOut" }}
              />

              {steps.map((step, index) => {
                const x = 110 + index * 200;
                const y = 170;
                const delay = 0.12 + index * 0.12;
                const descriptionLines = wrapSvgText(step.description, 28);

                return (
                  <motion.g
                    key={step.title}
                    initial={{ opacity: 0, y: 12 }}
                    animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 12 }}
                    transition={{ duration: 0.7, delay }}
                  >
                    <rect
                      x={x - 90}
                      y={y - 55}
                      width="180"
                      height="110"
                      rx="20"
                      fill="rgba(255,255,255,0.04)"
                      stroke="rgba(255,255,255,0.14)"
                    />
                    <circle
                      cx={x - 62}
                      cy={y - 22}
                      r="10"
                      fill="rgba(255,255,255,0.18)"
                    />
                    <text
                      x={x - 45}
                      y={y - 18}
                      fontSize="14"
                      fontWeight="600"
                      fill="rgba(255,255,255,0.92)"
                    >
                      {step.title}
                    </text>
                    <text
                      x={x - 62}
                      y={y + 12}
                      fontSize="12"
                      fill="rgba(255,255,255,0.7)"
                    >
                      {descriptionLines.map((line, lineIndex) => (
                        <tspan
                          key={`${step.title}-line-${lineIndex}`}
                          x={x - 62}
                          dy={lineIndex === 0 ? 0 : 14}
                        >
                          {line}
                        </tspan>
                      ))}
                    </text>
                  </motion.g>
                );
              })}
            </motion.svg>
          </div>
        </div>
      </div>
    </section>
  );
}
