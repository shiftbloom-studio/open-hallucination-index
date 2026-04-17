"use client";

import { motion, useInView, useAnimationFrame } from "framer-motion";
import { useRef, useState, useEffect } from "react";
import { cn } from "@/lib/utils";

type StepColor = "indigo" | "info" | "danger" | "warning" | "success";

const steps: {
  title: string;
  description: string;
  icon: string;
  color: StepColor;
}[] = [
  { title: "L1 · Decompose", description: "Atomic claims + multi-source evidence", icon: "🔬", color: "indigo" },
  { title: "L2 · Route", description: "5 domains · weighted assignment", icon: "🧭", color: "info" },
  // L3 NLI highlighted in red per reference image and brand canon.
  { title: "L3 · NLI", description: "Gemini 3 Pro cross-encoder", icon: "⚖️", color: "danger" },
  { title: "L4 · PCG", description: "TRW-BP belief propagation", icon: "🕸️", color: "warning" },
  { title: "L5 · Conformal", description: "Calibrated intervals @ 90% coverage", icon: "🎯", color: "success" },
];

const colorMap: Record<StepColor, { bg: string; fg: string; softBg: string }> = {
  indigo: { bg: "var(--brand-indigo)", fg: "var(--brand-indigo-strong)", softBg: "rgba(99,102,241,0.12)" },
  info: { bg: "var(--brand-info)", fg: "var(--brand-info)", softBg: "rgba(2,132,199,0.12)" },
  danger: { bg: "var(--brand-danger)", fg: "var(--brand-danger)", softBg: "rgba(230,57,70,0.12)" },
  warning: { bg: "var(--brand-warning)", fg: "var(--brand-warning)", softBg: "rgba(217,119,6,0.12)" },
  success: { bg: "var(--brand-success)", fg: "var(--brand-success)", softBg: "rgba(5,150,105,0.12)" },
};

function DataPulse({ delay, duration }: { delay: number; duration: number }) {
  return (
    <motion.div
      className="absolute top-1/2 -translate-y-1/2 h-1.5 w-5 rounded-full"
      style={{
        background:
          "linear-gradient(90deg, var(--brand-indigo), var(--brand-danger))",
        boxShadow: "0 0 10px rgba(99,102,241,0.55)",
      }}
      initial={{ left: "0%", opacity: 0 }}
      animate={{ left: ["0%", "100%"], opacity: [0, 1, 1, 0] }}
      transition={{ duration, delay, repeat: Infinity, ease: "linear" }}
    />
  );
}

function ConnectionLine({ isActive }: { isActive: boolean }) {
  return (
    <div className="relative flex-1 h-0.5 mx-1 overflow-hidden">
      <div className="absolute inset-0 bg-[color:var(--border-subtle)] rounded-full" />
      <motion.div
        className="absolute inset-0 rounded-full"
        style={{
          background:
            "linear-gradient(90deg, rgba(99,102,241,0.7), rgba(230,57,70,0.7))",
        }}
        initial={{ scaleX: 0, originX: 0 }}
        animate={{ scaleX: isActive ? 1 : 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      />
      {isActive && (
        <>
          <DataPulse delay={0} duration={2} />
          <DataPulse delay={0.7} duration={2} />
          <DataPulse delay={1.4} duration={2} />
        </>
      )}
    </div>
  );
}

function StepCard({
  step,
  index,
  isActive,
  isCompleted,
  onClick,
}: {
  step: (typeof steps)[number];
  index: number;
  isActive: boolean;
  isCompleted: boolean;
  onClick: () => void;
}) {
  const pathRef = useRef<SVGRectElement>(null);
  const progress = useRef(0);
  const [borderPosition, setBorderPosition] = useState({ x: 0, y: 0 });
  const c = colorMap[step.color];

  useAnimationFrame(() => {
    if (!isActive || !pathRef.current) return;
    const rect = pathRef.current;
    const w = rect.width.baseVal.value;
    const h = rect.height.baseVal.value;
    const perimeter = 2 * (w + h);
    const speed = 0.05;
    progress.current = (progress.current + speed) % perimeter;

    let x = 0,
      y = 0;
    const p = progress.current;
    if (p < w) {
      x = p;
      y = 0;
    } else if (p < w + h) {
      x = w;
      y = p - w;
    } else if (p < 2 * w + h) {
      x = w - (p - w - h);
      y = h;
    } else {
      x = 0;
      y = h - (p - 2 * w - h);
    }
    setBorderPosition({ x, y });
  });

  return (
    <motion.button
      onClick={onClick}
      className={cn(
        "relative group flex flex-col items-center p-3 rounded-xl transition-all duration-300 cursor-pointer",
        "border border-[color:var(--border-subtle)] bg-surface-elevated",
      )}
      style={
        isActive
          ? { borderColor: c.fg, boxShadow: `0 0 0 3px ${c.softBg}` }
          : isCompleted
            ? { borderColor: "var(--brand-success)", background: "rgba(5,150,105,0.04)" }
            : undefined
      }
      initial={{ opacity: 0, y: 20, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      whileHover={{ scale: 1.02, y: -2 }}
      whileTap={{ scale: 0.98 }}
    >
      {isActive && (
        <svg className="absolute inset-0 w-full h-full pointer-events-none overflow-visible">
          <rect
            ref={pathRef}
            x="0"
            y="0"
            width="100%"
            height="100%"
            rx="12"
            fill="none"
            className="opacity-0"
          />
          <circle
            cx={borderPosition.x}
            cy={borderPosition.y}
            r="18"
            fill="url(#glowGradient)"
            className="blur-sm"
          />
          <defs>
            <radialGradient id="glowGradient">
              <stop offset="0%" stopColor={c.bg} stopOpacity="0.55" />
              <stop offset="100%" stopColor={c.bg} stopOpacity="0" />
            </radialGradient>
          </defs>
        </svg>
      )}

      <div
        className="absolute -top-2 -right-2 w-5 h-5 rounded-full text-[10px] font-bold flex items-center justify-center border"
        style={
          isActive
            ? { background: c.bg, borderColor: c.bg, color: "#ffffff" }
            : isCompleted
              ? { background: "var(--brand-success)", borderColor: "var(--brand-success)", color: "#ffffff" }
              : { background: "var(--surface-elevated)", borderColor: "var(--border-default)", color: "var(--brand-muted)" }
        }
      >
        {isCompleted ? "✓" : index + 1}
      </div>

      <motion.div
        className="w-10 h-10 rounded-lg flex items-center justify-center text-lg mb-2 shadow-sm"
        style={{
          background: c.softBg,
          color: c.fg,
        }}
        animate={isActive ? { scale: [1, 1.08, 1] } : {}}
        transition={{ duration: 1.5, repeat: isActive ? Infinity : 0 }}
      >
        {step.icon}
      </motion.div>

      <span className="text-xs font-semibold text-brand-ink mb-0.5">
        {step.title}
      </span>

      <span className="text-[10px] text-brand-muted text-center leading-tight">
        {step.description}
      </span>

      {isActive && (
        <motion.div
          className="absolute inset-0 rounded-xl border-2 pointer-events-none"
          style={{ borderColor: c.fg, opacity: 0.35 }}
          animate={{ opacity: [0.35, 0, 0.35] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      )}
    </motion.button>
  );
}

export function ArchitectureFlow() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-10% 0px" });
  const [activeStep, setActiveStep] = useState(0);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    if (!isInView || isHovered) return;
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % steps.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [isInView, isHovered]);

  return (
    <section ref={ref} className="relative w-full py-12 md:py-16">
      <div className="mx-auto max-w-5xl px-4">
        <motion.div
          className="text-center mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
        >
          <span className="label-mono" style={{ color: "var(--brand-indigo-strong)" }}>
            Architecture
          </span>
          <h2
            className="mt-2 font-display font-semibold tracking-tight text-brand-ink"
            style={{ fontSize: "clamp(1.5rem, 3.25vw, 2.5rem)", lineHeight: 1.1 }}
          >
            The v2 verification pipeline
          </h2>
          <p className="mt-2 text-sm text-brand-muted max-w-md mx-auto md:text-base">
            Each layer is independently scored and cached. Degradation is surfaced per-claim via{" "}
            <span className="font-mono text-brand-ink">fallback_used</span>.
          </p>
        </motion.div>

        <motion.div
          className="relative flex items-center justify-between gap-1 rounded-2xl border border-[color:var(--border-subtle)] bg-surface-elevated p-4 shadow-sm"
          initial={{ opacity: 0, scale: 0.97 }}
          animate={isInView ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 0.8, delay: 0.2 }}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          <div
            className="absolute inset-0 rounded-2xl pointer-events-none"
            style={{
              background:
                "linear-gradient(90deg, rgba(99,102,241,0.04) 0%, transparent 50%, rgba(230,57,70,0.04) 100%)",
            }}
          />

          {steps.map((step, index) => (
            <div key={step.title} className="contents">
              <StepCard
                step={step}
                index={index}
                isActive={activeStep === index}
                isCompleted={activeStep > index}
                onClick={() => setActiveStep(index)}
              />
              {index < steps.length - 1 && (
                <ConnectionLine isActive={isInView && activeStep > index} />
              )}
            </div>
          ))}
        </motion.div>

        <motion.div
          className="flex justify-center gap-1.5 mt-4"
          initial={{ opacity: 0 }}
          animate={isInView ? { opacity: 1 } : {}}
          transition={{ delay: 0.5 }}
        >
          {steps.map((_, index) => (
            <button
              key={index}
              onClick={() => setActiveStep(index)}
              aria-label={`Go to step ${index + 1}`}
              className={cn(
                "h-1.5 rounded-full transition-all duration-300",
                activeStep === index
                  ? "w-4 bg-brand-indigo"
                  : "w-1.5 bg-[color:var(--border-default)] hover:bg-brand-subtle",
              )}
            />
          ))}
        </motion.div>
      </div>
    </section>
  );
}
