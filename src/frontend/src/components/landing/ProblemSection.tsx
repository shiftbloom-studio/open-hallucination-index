"use client";

import { motion, useInView, useSpring, useTransform, AnimatePresence } from "framer-motion";
import { useRef, useEffect, useState } from "react";

function AnimatedCounter({ value, suffix = "" }: { value: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true });
  const spring = useSpring(0, { duration: 2000 });
  const display = useTransform(spring, (v) => Math.floor(v));
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    if (isInView) spring.set(value);
  }, [isInView, spring, value]);

  useEffect(() => display.on("change", (v) => setDisplayValue(v)), [display]);

  return (
    <span ref={ref} className="num-mono">
      {displayValue}{suffix}
    </span>
  );
}

// Hallucinated vs Verified claims for the animation
const claims = [
  {
    hallucinated: "The Eiffel Tower was built in 1920",
    verified: "The Eiffel Tower was built in 1889",
    wrongWord: "1920",
    isTrue: false,
  },
  {
    hallucinated: "Water boils at 100°C at sea level",
    verified: "Water boils at 100°C at sea level",
    wrongWord: null,
    isTrue: true,
  },
  {
    hallucinated: "Shakespeare wrote 47 plays",
    verified: "Shakespeare wrote ~37 plays",
    wrongWord: "47",
    isTrue: false,
  },
  {
    hallucinated: "The human body has 206 bones",
    verified: "The human body has 206 bones",
    wrongWord: null,
    isTrue: true,
  },
  {
    hallucinated: "Einstein discovered gravity in 1687",
    verified: "Newton discovered gravity in 1687",
    wrongWord: "Einstein",
    isTrue: false,
  },
  {
    hallucinated: "Mars has 3 moons orbiting it",
    verified: "Mars has 2 moons orbiting it",
    wrongWord: "3",
    isTrue: false,
  },
  {
    hallucinated: "The speed of light is ~300,000 km/s",
    verified: "The speed of light is ~300,000 km/s",
    wrongWord: null,
    isTrue: true,
  },
  {
    hallucinated: "The Amazon is 8,000 km long",
    verified: "The Amazon is ~6,400 km long",
    wrongWord: "8,000",
    isTrue: false,
  },
];

const glitchChars = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~αβγδεζηθ";

function GlitchText({ text, isGlitching }: { text: string; isGlitching: boolean }) {
  const [displayText, setDisplayText] = useState(text);

  useEffect(() => {
    if (!isGlitching) {
      setDisplayText(text);
      return;
    }

    const interval = setInterval(() => {
      setDisplayText(
        text
          .split("")
          .map((char) => {
            if (char === " ") return " ";
            if (Math.random() > 0.7) {
              return glitchChars[Math.floor(Math.random() * glitchChars.length)];
            }
            return char;
          })
          .join(""),
      );
    }, 50);

    return () => clearInterval(interval);
  }, [text, isGlitching]);

  return <span>{displayText}</span>;
}

function HighlightedClaimText({
  text,
  wrongWord,
  showHighlight,
}: {
  text: string;
  wrongWord: string | null;
  showHighlight: boolean;
}) {
  if (!wrongWord || !showHighlight) {
    return <span>{text}</span>;
  }

  const parts = text.split(wrongWord);
  if (parts.length === 1) {
    return <span>{text}</span>;
  }

  return (
    <span>
      {parts[0]}
      <motion.span
        className="relative inline-block"
        // fix: framer-motion cannot animate FROM the CSS keyword "transparent".
        // Starting with the same rgba at alpha 0 is the supported form.
        initial={{ backgroundColor: "rgba(230,57,70,0)" }}
        animate={{ backgroundColor: "rgba(230,57,70,0.18)" }}
        transition={{ duration: 0.3 }}
      >
        <span className="relative font-semibold text-[color:var(--brand-danger)]">
          {wrongWord}
          <motion.span
            className="absolute -bottom-0.5 left-0 right-0 h-0.5"
            style={{ background: "var(--brand-danger)" }}
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          />
        </span>
      </motion.span>
      {parts.slice(1).join(wrongWord)}
    </span>
  );
}

function ScannerBeam({ isActive }: { isActive: boolean }) {
  return (
    <motion.div
      className="absolute left-0 right-0 h-0.5 pointer-events-none z-20"
      style={{
        background:
          "linear-gradient(90deg, transparent, rgba(99,102,241,0.8), rgba(230,57,70,0.6), transparent)",
        boxShadow: "0 0 20px 4px rgba(99,102,241,0.35), 0 0 40px 8px rgba(230,57,70,0.15)",
      }}
      initial={{ top: "0%", opacity: 0 }}
      animate={
        isActive
          ? { top: ["0%", "100%", "0%"], opacity: [0, 1, 1, 0] }
          : { opacity: 0 }
      }
      transition={{ duration: 2, ease: "easeInOut" }}
    />
  );
}

const PARTICLE_POSITIONS = [12, 28, 45, 62, 78, 88, 35, 55];

function VerificationParticle({ delay, verified, index }: { delay: number; verified: boolean; index: number }) {
  const base = verified ? "rgba(5,150,105,1)" : "rgba(230,57,70,1)";
  return (
    <motion.div
      className="absolute h-1.5 w-1.5 rounded-full"
      style={{
        left: `${PARTICLE_POSITIONS[index % PARTICLE_POSITIONS.length]}%`,
        background: verified ? "var(--brand-success)" : "var(--brand-danger)",
      }}
      initial={{ top: "50%", opacity: 0, scale: 0 }}
      animate={{
        top: verified ? "-20%" : "120%",
        opacity: [0, 1, 1, 0],
        scale: [0, 1, 1, 0],
        boxShadow: [
          `0 0 8px ${base.replace("1)", "0.8)")}`,
          `0 0 12px ${base.replace("1)", "0.9)")}`,
          `0 0 8px ${base.replace("1)", "0.8)")}`,
        ],
      }}
      transition={{ duration: 2, delay, repeat: Infinity, repeatDelay: 3 }}
    />
  );
}

function NeuralConnection({ startX, startY, endX, endY, delay }: {
  startX: number; startY: number; endX: number; endY: number; delay: number;
}) {
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none">
      <motion.path
        d={`M ${startX} ${startY} Q ${(startX + endX) / 2} ${startY - 20} ${endX} ${endY}`}
        fill="none"
        stroke="url(#neuralGradient)"
        strokeWidth="1"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: [0, 1, 0], opacity: [0, 0.6, 0] }}
        transition={{ duration: 2, delay, repeat: Infinity, repeatDelay: 2 }}
      />
      <defs>
        <linearGradient id="neuralGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="rgba(99,102,241,0.8)" />
          <stop offset="50%" stopColor="rgba(230,57,70,0.7)" />
          <stop offset="100%" stopColor="rgba(5,150,105,0.8)" />
        </linearGradient>
      </defs>
    </svg>
  );
}

type AnimationPhase = "analyzing" | "scanning" | "detected" | "verified";

interface AnimationState {
  claimIndex: number;
  phase: AnimationPhase;
}

function useAnimationState(isInView: boolean) {
  const [state, setState] = useState<AnimationState>({
    claimIndex: 0,
    phase: "analyzing",
  });

  const animationRef = useRef<{
    timeoutId: NodeJS.Timeout | null;
  }>({ timeoutId: null });

  useEffect(() => {
    if (!isInView) return;

    const PHASE_DURATIONS = {
      analyzing: 1500,
      scanning: 1800,
      detected: 1500,
      verified: 2000,
    };

    const scheduleNextPhase = (currentPhase: AnimationPhase, claimIndex: number) => {
      if (animationRef.current.timeoutId) {
        clearTimeout(animationRef.current.timeoutId);
      }

      const currentClaim = claims[claimIndex];

      const getNextPhaseData = (): { phase: AnimationPhase; claimIndex: number; delay: number } => {
        switch (currentPhase) {
          case "analyzing":
            return { phase: "scanning", claimIndex, delay: PHASE_DURATIONS.analyzing };
          case "scanning":
            if (currentClaim.isTrue) {
              return { phase: "verified", claimIndex, delay: PHASE_DURATIONS.scanning };
            }
            return { phase: "detected", claimIndex, delay: PHASE_DURATIONS.scanning };
          case "detected":
            return { phase: "verified", claimIndex, delay: PHASE_DURATIONS.detected };
          case "verified":
            return { phase: "analyzing", claimIndex: (claimIndex + 1) % claims.length, delay: PHASE_DURATIONS.verified };
        }
      };

      const { phase: nextPhase, claimIndex: nextClaimIndex, delay } = getNextPhaseData();

      animationRef.current.timeoutId = setTimeout(() => {
        setState({ phase: nextPhase, claimIndex: nextClaimIndex });
        scheduleNextPhase(nextPhase, nextClaimIndex);
      }, delay);
    };

    setState({ phase: "analyzing", claimIndex: 0 });
    scheduleNextPhase("analyzing", 0);

    const currentAnimation = animationRef.current;

    return () => {
      if (currentAnimation.timeoutId) {
        clearTimeout(currentAnimation.timeoutId);
      }
    };
  }, [isInView]);

  return state;
}

function HallucinationVisualizer({ isInView }: { isInView: boolean }) {
  const { claimIndex, phase } = useAnimationState(isInView);
  const claim = claims[claimIndex];

  return (
    <div
      className="relative w-full h-full min-h-[200px] overflow-hidden rounded-xl border border-[color:var(--border-subtle)] p-4"
      style={{ background: "var(--surface-elevated)" }}
    >
      {/* Grid */}
      <div
        className="absolute inset-0 opacity-40"
        style={{
          backgroundImage: `
            linear-gradient(rgba(99,102,241,0.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99,102,241,0.08) 1px, transparent 1px)
          `,
          backgroundSize: "20px 20px",
        }}
      />

      {[...Array(8)].map((_, i) => (
        <VerificationParticle key={i} delay={i * 0.3} verified={i % 2 === 0} index={i} />
      ))}

      <NeuralConnection startX={20} startY={180} endX={180} endY={40} delay={0} />
      <NeuralConnection startX={280} startY={40} endX={380} endY={180} delay={0.5} />

      <ScannerBeam isActive={phase === "scanning"} />

      <div className="relative z-10 flex flex-col items-center justify-center h-full px-2 sm:px-4">
        <motion.div className="flex items-center gap-2 mb-3 sm:mb-4" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <motion.div
            className="h-2 w-2 rounded-full"
            style={{
              background:
                phase === "analyzing"
                  ? "var(--brand-indigo)"
                  : phase === "scanning"
                    ? "var(--brand-info)"
                    : phase === "detected"
                      ? "var(--brand-danger)"
                      : "var(--brand-success)",
            }}
            animate={{
              scale: phase === "scanning" ? [1, 1.5, 1] : 1,
              boxShadow:
                phase === "scanning"
                  ? ["0 0 0px rgba(2,132,199,0.5)", "0 0 14px rgba(2,132,199,0.8)", "0 0 0px rgba(2,132,199,0.5)"]
                  : "none",
            }}
            transition={{ duration: 0.5, repeat: phase === "scanning" ? Infinity : 0 }}
          />
          <span
            className="label-mono"
            style={{
              color:
                phase === "analyzing"
                  ? "var(--brand-indigo)"
                  : phase === "scanning"
                    ? "var(--brand-info)"
                    : phase === "detected"
                      ? "var(--brand-danger)"
                      : "var(--brand-success)",
            }}
          >
            {phase === "analyzing"
              ? "Analyzing Claim..."
              : phase === "scanning"
                ? "Verifying..."
                : phase === "detected"
                  ? "Hallucination Detected!"
                  : claim.isTrue
                    ? "Verified"
                    : "Corrected"}
          </span>
        </motion.div>

        <motion.div
          className="relative rounded-lg border px-3 py-3 sm:px-6 sm:py-4 transition-all duration-500 max-w-full"
          style={{
            background:
              phase === "analyzing"
                ? "rgba(99,102,241,0.06)"
                : phase === "scanning"
                  ? "rgba(2,132,199,0.06)"
                  : phase === "detected"
                    ? "rgba(230,57,70,0.06)"
                    : "rgba(5,150,105,0.06)",
            borderColor:
              phase === "analyzing"
                ? "rgba(99,102,241,0.35)"
                : phase === "scanning"
                  ? "rgba(2,132,199,0.35)"
                  : phase === "detected"
                    ? "rgba(230,57,70,0.35)"
                    : "rgba(5,150,105,0.35)",
          }}
          animate={{ x: phase === "detected" ? [0, -2, 2, -1, 1, 0] : 0 }}
          transition={{ duration: 0.3, repeat: phase === "detected" ? 3 : 0, repeatDelay: 0.1 }}
        >
          {phase === "detected" && (
            <motion.div
              className="absolute inset-0 rounded-lg"
              style={{ background: "rgba(230,57,70,0.08)" }}
              animate={{ opacity: [0, 0.4, 0] }}
              transition={{ duration: 0.2, repeat: 3 }}
            />
          )}

          <p className="text-xs sm:text-sm md:text-base text-brand-ink text-center font-mono leading-relaxed">
            <AnimatePresence mode="wait">
              {phase === "verified" ? (
                <motion.span
                  key="verified"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  style={{ color: "var(--brand-success)" }}
                >
                  {claim.verified}
                </motion.span>
              ) : phase === "detected" ? (
                <motion.span key="detected" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <HighlightedClaimText text={claim.hallucinated} wrongWord={claim.wrongWord} showHighlight={true} />
                </motion.span>
              ) : (
                <motion.span key="analyzing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <GlitchText text={claim.hallucinated} isGlitching={phase === "analyzing"} />
                </motion.span>
              )}
            </AnimatePresence>
          </p>
        </motion.div>

        <motion.div
          className="mt-4 sm:mt-6 w-full max-w-[180px] sm:max-w-[200px]"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex justify-between text-[9px] sm:text-[10px] text-brand-subtle mb-1">
            <span>0%</span>
            <span className="text-brand-muted font-medium">Trust Score</span>
            <span>100%</span>
          </div>
          <div className="h-1 sm:h-1.5 bg-[color:var(--border-subtle)] rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{
                background:
                  phase === "verified"
                    ? "var(--brand-success)"
                    : phase === "detected"
                      ? "var(--brand-danger)"
                      : "var(--brand-indigo)",
              }}
              initial={{ width: "0%" }}
              animate={{
                width:
                  phase === "analyzing"
                    ? "50%"
                    : phase === "scanning"
                      ? "50%"
                      : phase === "detected"
                        ? "5%"
                        : "100%",
              }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            />
          </div>
        </motion.div>

        <div className="absolute bottom-2 right-2 sm:right-3 flex gap-0.5 sm:gap-1">
          {claims.map((_, i) => (
            <motion.div
              key={i}
              className="h-1 w-1 rounded-full"
              style={{
                background: i === claimIndex ? "var(--brand-indigo)" : "var(--border-default)",
              }}
              animate={i === claimIndex ? { scale: [1, 1.3, 1] } : {}}
              transition={{ duration: 1, repeat: Infinity }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

const stats = [
  { value: 27, suffix: "%", label: "Avg. hallucination rate" },
  { value: 99, suffix: "%", label: "Detection accuracy" },
  { value: 50, suffix: "ms", label: "Verification latency" },
];

export function ProblemSection() {
  const ref = useRef<HTMLDivElement | null>(null);
  const isInView = useInView(ref, { once: true, margin: "-10% 0px" });

  return (
    <section className="relative w-full">
      <div className="mx-auto max-w-7xl px-4 py-16 md:py-24">
        <motion.div
          ref={ref}
          className="relative overflow-hidden rounded-3xl border border-[color:var(--border-subtle)] px-6 py-12 md:px-14"
          style={{ background: "var(--surface-soft)" }}
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          {/* Subtle corner accents */}
          <motion.div
            className="absolute top-0 left-0 h-40 w-40 rounded-full"
            style={{ background: "var(--brand-indigo)", opacity: 0.08, filter: "blur(80px)" }}
            animate={{ scale: [1, 1.2, 1], opacity: [0.06, 0.1, 0.06] }}
            transition={{ duration: 6, repeat: Infinity }}
          />
          <motion.div
            className="absolute bottom-0 right-0 h-48 w-48 rounded-full"
            style={{ background: "var(--brand-danger)", opacity: 0.06, filter: "blur(90px)" }}
            animate={{ scale: [1.2, 1, 1.2], opacity: [0.04, 0.08, 0.04] }}
            transition={{ duration: 7, repeat: Infinity }}
          />

          <div className="relative z-10 grid gap-10 lg:grid-cols-2 lg:items-center">
            <div>
              <motion.p
                initial={{ opacity: 0, x: -20 }}
                animate={isInView ? { opacity: 1, x: 0 } : {}}
                transition={{ duration: 0.6 }}
                className="label-mono"
                style={{ color: "var(--brand-indigo-strong)" }}
              >
                The Problem
              </motion.p>

              <motion.h2
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.7, delay: 0.1 }}
                className="mt-4 font-display font-semibold tracking-tight text-brand-ink"
                style={{ fontSize: "clamp(1.75rem, 4vw, 3rem)", lineHeight: 1.08, letterSpacing: "-0.03em" }}
              >
                LLM confidence is theater.{" "}
                <motion.span
                  className="display-accent"
                  animate={isInView ? { opacity: [0.7, 1, 0.7] } : {}}
                  transition={{ duration: 2.5, repeat: Infinity }}
                >
                  Calibration isn&apos;t.
                </motion.span>
              </motion.h2>

              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.7, delay: 0.2 }}
                className="mt-6 max-w-xl text-base leading-relaxed text-brand-muted md:text-lg"
              >
                Ask an LLM how sure it is and it&apos;ll say 95%. Ask again — still 95%. Ask after
                it&apos;s wrong — still 95%. Self-reported confidence is uncorrelated with truth.
                OHI&apos;s 0.85 [0.78, 0.91] at 90% coverage is a guarantee, not a vibe.
              </motion.p>

              <motion.div
                className="grid grid-cols-3 gap-4 mt-8"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={isInView ? { opacity: 1, scale: 1 } : {}}
                transition={{ duration: 0.8, delay: 0.3 }}
              >
                {stats.map((stat, index) => (
                  <motion.div
                    key={stat.label}
                    className="relative flex flex-col items-center rounded-xl border border-[color:var(--border-subtle)] bg-surface-elevated p-4 shadow-sm transition-colors"
                    initial={{ opacity: 0, y: 20 }}
                    animate={isInView ? { opacity: 1, y: 0 } : {}}
                    transition={{ delay: 0.4 + index * 0.1 }}
                    whileHover={{ scale: 1.03, borderColor: "rgba(99,102,241,0.5)" }}
                  >
                    <span className="text-2xl font-display font-semibold text-brand-ink md:text-3xl">
                      <AnimatedCounter value={stat.value} suffix={stat.suffix} />
                    </span>
                    <span className="mt-1 text-[10px] text-brand-muted text-center md:text-xs">
                      {stat.label}
                    </span>
                  </motion.div>
                ))}
              </motion.div>
            </div>

            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={isInView ? { opacity: 1, scale: 1 } : {}}
              transition={{ duration: 0.8, delay: 0.4 }}
            >
              <HallucinationVisualizer isInView={isInView} />
            </motion.div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
