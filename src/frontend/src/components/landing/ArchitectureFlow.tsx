"use client";

import { motion, useInView, useAnimationFrame } from "framer-motion";
import { useRef, useState, useEffect } from "react";
import { cn } from "@/lib/utils";

const steps = [
  {
    title: "Input",
    description: "Prompt enters pipeline",
    icon: "📝",
    color: "from-violet-500 to-purple-600",
  },
  {
    title: "Decompose",
    description: "Atomize into claims",
    icon: "🔬",
    color: "from-blue-500 to-cyan-500",
  },
  {
    title: "Verify",
    description: "Check against sources",
    icon: "✓",
    color: "from-emerald-500 to-teal-500",
  },
  {
    title: "Score",
    description: "Output trust metric",
    icon: "📊",
    color: "from-amber-500 to-orange-500",
  },
] as const;

function DataPulse({ delay, duration }: { delay: number; duration: number }) {
  return (
    <motion.div
      className="absolute top-1/2 -translate-y-1/2 h-1.5 w-6 rounded-full bg-gradient-to-r from-violet-500 via-cyan-400 to-emerald-400 shadow-[0_0_12px_rgba(139,92,246,0.8)]"
      initial={{ left: "0%", opacity: 0 }}
      animate={{
        left: ["0%", "100%"],
        opacity: [0, 1, 1, 0],
      }}
      transition={{
        duration,
        delay,
        repeat: Infinity,
        ease: "linear",
      }}
    />
  );
}

function ConnectionLine({ isActive }: { isActive: boolean }) {
  return (
    <div className="relative flex-1 h-0.5 mx-1 overflow-hidden">
      {/* Base line */}
      <div className="absolute inset-0 bg-white/10 rounded-full" />
      
      {/* Glowing active line */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-violet-500/50 via-cyan-400/50 to-emerald-400/50 rounded-full"
        initial={{ scaleX: 0, originX: 0 }}
        animate={{ scaleX: isActive ? 1 : 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      />

      {/* Data pulses */}
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
        "bg-white/[0.03] border border-white/10 backdrop-blur-sm",
        isActive && "border-white/20 bg-white/[0.06]",
        isCompleted && "border-emerald-500/30 bg-emerald-500/[0.05]"
      )}
      initial={{ opacity: 0, y: 20, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      whileHover={{ scale: 1.02, y: -2 }}
      whileTap={{ scale: 0.98 }}
    >
      {/* Moving border glow for active state */}
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
            r="20"
            fill="url(#glowGradient)"
            className="blur-sm"
          />
          <defs>
            <radialGradient id="glowGradient">
              <stop offset="0%" stopColor="rgba(139,92,246,0.8)" />
              <stop offset="100%" stopColor="transparent" />
            </radialGradient>
          </defs>
        </svg>
      )}

      {/* Step number badge */}
      <div
        className={cn(
          "absolute -top-2 -right-2 w-5 h-5 rounded-full text-[10px] font-bold flex items-center justify-center",
          "bg-neutral-800 border border-white/20 text-white/70",
          isActive && "bg-violet-600 border-violet-400 text-white",
          isCompleted && "bg-emerald-600 border-emerald-400 text-white"
        )}
      >
        {isCompleted ? "✓" : index + 1}
      </div>

      {/* Icon */}
      <motion.div
        className={cn(
          "w-10 h-10 rounded-lg flex items-center justify-center text-lg mb-2",
          "bg-gradient-to-br",
          step.color,
          "shadow-lg",
          isActive && "shadow-violet-500/30"
        )}
        animate={isActive ? { scale: [1, 1.1, 1] } : {}}
        transition={{ duration: 1.5, repeat: isActive ? Infinity : 0 }}
      >
        {step.icon}
      </motion.div>

      {/* Title */}
      <span className="text-xs font-semibold text-white/90 mb-0.5">
        {step.title}
      </span>

      {/* Description */}
      <span className="text-[10px] text-white/50 text-center leading-tight">
        {step.description}
      </span>

      {/* Active indicator pulse */}
      {isActive && (
        <motion.div
          className="absolute inset-0 rounded-xl border-2 border-violet-500/50"
          animate={{ opacity: [0.5, 0, 0.5] }}
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

  // Auto-advance through steps
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
        {/* Header */}
        <motion.div
          className="text-center mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
        >
          <span className="text-xs font-medium tracking-widest text-violet-400 uppercase">
            Architecture
          </span>
          <h2 className="mt-2 text-2xl md:text-4xl font-bold bg-gradient-to-r from-white via-white to-white/60 bg-clip-text text-transparent">
            Verification Pipeline
          </h2>
          <p className="mt-2 text-sm text-neutral-400 max-w-md mx-auto">
            Every generation flows through four stages of verification
          </p>
        </motion.div>

        {/* Pipeline visualization */}
        <motion.div
          className="relative flex items-center justify-between gap-1 p-4 rounded-2xl bg-black/40 border border-white/10 backdrop-blur-xl"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={isInView ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 0.8, delay: 0.2 }}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          {/* Background gradient */}
          <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-violet-600/5 via-transparent to-emerald-600/5 pointer-events-none" />

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

        {/* Auto-advance indicator */}
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
                "w-1.5 h-1.5 rounded-full transition-all duration-300",
                activeStep === index
                  ? "w-4 bg-violet-500"
                  : "bg-white/20 hover:bg-white/40"
              )}
            />
          ))}
        </motion.div>
      </div>
    </section>
  );
}
