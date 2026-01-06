"use client";

import { motion, useInView, useSpring, useTransform, AnimatePresence } from "framer-motion";
import { useRef, useEffect, useState, useCallback } from "react";

function AnimatedCounter({ value, suffix = "" }: { value: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true });
  const spring = useSpring(0, { duration: 2000 });
  const display = useTransform(spring, (v) => Math.floor(v));
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    if (isInView) spring.set(value);
  }, [isInView, spring, value]);

  useEffect(() => {
    return display.on("change", (v) => setDisplayValue(v));
  }, [display]);

  return (
    <span ref={ref} className="tabular-nums">
      {displayValue}{suffix}
    </span>
  );
}

// Hallucinated vs Verified claims for the animation
const claims = [
  { 
    hallucinated: "The Eiffel Tower was built in 1920",
    verified: "The Eiffel Tower was built in 1889",
    isTrue: false 
  },
  { 
    hallucinated: "Water boils at 100°C at sea level",
    verified: "Water boils at 100°C at sea level",
    isTrue: true 
  },
  { 
    hallucinated: "Shakespeare wrote 47 plays",
    verified: "Shakespeare wrote ~37 plays",
    isTrue: false 
  },
];

// Glitch character set for the hallucination effect
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
          .map((char, i) => {
            if (char === " ") return " ";
            if (Math.random() > 0.7) {
              return glitchChars[Math.floor(Math.random() * glitchChars.length)];
            }
            return char;
          })
          .join("")
      );
    }, 50);
    
    return () => clearInterval(interval);
  }, [text, isGlitching]);
  
  return <span>{displayText}</span>;
}

// Verification scanner beam
function ScannerBeam({ isActive }: { isActive: boolean }) {
  return (
    <motion.div
      className="absolute left-0 right-0 h-0.5 pointer-events-none z-20"
      style={{
        background: "linear-gradient(90deg, transparent, rgba(34,211,238,0.8), rgba(167,139,250,0.8), transparent)",
        boxShadow: "0 0 20px 4px rgba(34,211,238,0.5), 0 0 40px 8px rgba(167,139,250,0.3)",
      }}
      initial={{ top: "0%", opacity: 0 }}
      animate={isActive ? {
        top: ["0%", "100%", "0%"],
        opacity: [0, 1, 1, 0],
      } : { opacity: 0 }}
      transition={{ duration: 2, ease: "easeInOut" }}
    />
  );
}

// Floating verification particles
function VerificationParticle({ delay, verified }: { delay: number; verified: boolean }) {
  return (
    <motion.div
      className={`absolute w-1.5 h-1.5 rounded-full ${verified ? "bg-emerald-400" : "bg-red-400"}`}
      style={{
        left: `${Math.random() * 100}%`,
        boxShadow: verified 
          ? "0 0 8px rgba(52,211,153,0.8)" 
          : "0 0 8px rgba(248,113,113,0.8)",
      }}
      initial={{ top: "50%", opacity: 0, scale: 0 }}
      animate={{
        top: verified ? "-20%" : "120%",
        opacity: [0, 1, 1, 0],
        scale: [0, 1, 1, 0],
      }}
      transition={{
        duration: 2,
        delay,
        repeat: Infinity,
        repeatDelay: 3,
      }}
    />
  );
}

// Neural pathway connections
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
          <stop offset="0%" stopColor="rgba(167,139,250,0.8)" />
          <stop offset="50%" stopColor="rgba(34,211,238,0.8)" />
          <stop offset="100%" stopColor="rgba(52,211,153,0.8)" />
        </linearGradient>
      </defs>
    </svg>
  );
}

// Main hallucination visualization component
function HallucinationVisualizer({ isInView }: { isInView: boolean }) {
  const [currentClaim, setCurrentClaim] = useState(0);
  const [phase, setPhase] = useState<"hallucinating" | "scanning" | "verified">("hallucinating");
  
  useEffect(() => {
    if (!isInView) return;
    
    const cycle = () => {
      // Phase 1: Show hallucinated text with glitch
      setPhase("hallucinating");
      
      setTimeout(() => {
        // Phase 2: Scanning
        setPhase("scanning");
      }, 2000);
      
      setTimeout(() => {
        // Phase 3: Show verified result
        setPhase("verified");
      }, 4000);
      
      setTimeout(() => {
        // Move to next claim
        setCurrentClaim((prev) => (prev + 1) % claims.length);
      }, 6000);
    };
    
    cycle();
    const interval = setInterval(cycle, 6000);
    return () => clearInterval(interval);
  }, [isInView]);
  
  const claim = claims[currentClaim];
  
  return (
    <div className="relative w-full h-full min-h-[200px] overflow-hidden rounded-xl bg-black/40 border border-white/10 p-4">
      {/* Background grid pattern */}
      <div 
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage: `
            linear-gradient(rgba(139,92,246,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(139,92,246,0.1) 1px, transparent 1px)
          `,
          backgroundSize: "20px 20px",
        }}
      />
      
      {/* Floating particles */}
      {[...Array(8)].map((_, i) => (
        <VerificationParticle 
          key={i} 
          delay={i * 0.3} 
          verified={i % 2 === 0} 
        />
      ))}
      
      {/* Neural connections */}
      <NeuralConnection startX={20} startY={180} endX={180} endY={40} delay={0} />
      <NeuralConnection startX={280} startY={40} endX={380} endY={180} delay={0.5} />
      
      {/* Scanner beam */}
      <ScannerBeam isActive={phase === "scanning"} />
      
      {/* Main content area */}
      <div className="relative z-10 flex flex-col items-center justify-center h-full">
        {/* Status indicator */}
        <motion.div 
          className="flex items-center gap-2 mb-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <motion.div
            className={`w-2 h-2 rounded-full ${
              phase === "hallucinating" ? "bg-red-500" :
              phase === "scanning" ? "bg-cyan-400" :
              claim.isTrue ? "bg-emerald-500" : "bg-orange-500"
            }`}
            animate={{ 
              scale: phase === "scanning" ? [1, 1.5, 1] : 1,
              boxShadow: phase === "scanning" 
                ? ["0 0 0px rgba(34,211,238,0.5)", "0 0 20px rgba(34,211,238,0.8)", "0 0 0px rgba(34,211,238,0.5)"]
                : "none"
            }}
            transition={{ duration: 0.5, repeat: phase === "scanning" ? Infinity : 0 }}
          />
          <span className={`text-xs font-medium uppercase tracking-wider ${
            phase === "hallucinating" ? "text-red-400" :
            phase === "scanning" ? "text-cyan-400" :
            claim.isTrue ? "text-emerald-400" : "text-orange-400"
          }`}>
            {phase === "hallucinating" ? "Analyzing Output..." :
             phase === "scanning" ? "Verifying Claims..." :
             claim.isTrue ? "Verified ✓" : "Hallucination Detected"}
          </span>
        </motion.div>
        
        {/* Claim text box */}
        <motion.div
          className={`relative px-6 py-4 rounded-lg border backdrop-blur-sm transition-all duration-500 ${
            phase === "hallucinating" 
              ? "bg-red-500/10 border-red-500/30" 
              : phase === "scanning"
              ? "bg-cyan-500/10 border-cyan-500/30"
              : claim.isTrue 
              ? "bg-emerald-500/10 border-emerald-500/30"
              : "bg-orange-500/10 border-orange-500/30"
          }`}
          animate={{
            x: phase === "hallucinating" ? [0, -2, 2, -1, 1, 0] : 0,
          }}
          transition={{ 
            duration: 0.3, 
            repeat: phase === "hallucinating" ? Infinity : 0,
            repeatDelay: 0.1,
          }}
        >
          {/* Glitch lines overlay */}
          {phase === "hallucinating" && (
            <>
              <motion.div
                className="absolute inset-0 bg-red-500/5 rounded-lg"
                animate={{ opacity: [0, 0.3, 0] }}
                transition={{ duration: 0.15, repeat: Infinity }}
              />
              <motion.div
                className="absolute left-0 right-0 h-px bg-red-400/50"
                animate={{ top: ["0%", "100%"] }}
                transition={{ duration: 0.8, repeat: Infinity }}
              />
            </>
          )}
          
          <p className="text-sm md:text-base text-white/90 text-center font-mono">
            <AnimatePresence mode="wait">
              {phase === "verified" ? (
                <motion.span
                  key="verified"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                >
                  {claim.verified}
                </motion.span>
              ) : (
                <motion.span
                  key="hallucinated"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <GlitchText 
                    text={claim.hallucinated} 
                    isGlitching={phase === "hallucinating"} 
                  />
                </motion.span>
              )}
            </AnimatePresence>
          </p>
        </motion.div>
        
        {/* Trust score meter */}
        <motion.div 
          className="mt-6 w-full max-w-[200px]"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex justify-between text-[10px] text-white/50 mb-1">
            <span>0%</span>
            <span className="text-white/70 font-medium">Trust Score</span>
            <span>100%</span>
          </div>
          <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              className={`h-full rounded-full ${
                phase === "verified" && claim.isTrue 
                  ? "bg-gradient-to-r from-emerald-500 to-emerald-400"
                  : phase === "verified"
                  ? "bg-gradient-to-r from-orange-500 to-red-500"
                  : "bg-gradient-to-r from-violet-500 to-cyan-400"
              }`}
              initial={{ width: "0%" }}
              animate={{ 
                width: phase === "hallucinating" ? "30%" :
                       phase === "scanning" ? "60%" :
                       claim.isTrue ? "98%" : "23%"
              }}
              transition={{ duration: 1, ease: "easeOut" }}
            />
          </div>
        </motion.div>
        
        {/* Claim counter */}
        <div className="absolute bottom-2 right-3 flex gap-1">
          {claims.map((_, i) => (
            <motion.div
              key={i}
              className={`w-1 h-1 rounded-full ${
                i === currentClaim ? "bg-violet-400" : "bg-white/20"
              }`}
              animate={i === currentClaim ? { scale: [1, 1.3, 1] } : {}}
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
          className="relative overflow-hidden rounded-3xl border border-white/15 bg-slate-900/60 px-6 py-12 backdrop-blur-xl md:px-14"
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          {/* Animated gradient border */}
          <motion.div
            className="absolute inset-0 rounded-3xl opacity-60"
            style={{
              background: "linear-gradient(90deg, transparent, rgba(167,139,250,0.4), transparent)",
              backgroundSize: "200% 100%",
            }}
            animate={{ backgroundPosition: ["-100% 0%", "200% 0%"] }}
            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
          />
          
          {/* Pulsing corner accents */}
          <motion.div
            className="absolute top-0 left-0 w-32 h-32 bg-violet-500/20 rounded-full blur-3xl"
            animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
            transition={{ duration: 4, repeat: Infinity }}
          />
          <motion.div
            className="absolute bottom-0 right-0 w-40 h-40 bg-cyan-500/20 rounded-full blur-3xl"
            animate={{ scale: [1.2, 1, 1.2], opacity: [0.2, 0.4, 0.2] }}
            transition={{ duration: 5, repeat: Infinity }}
          />
          
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-white/15 via-transparent to-transparent" />

          <div className="relative z-10 grid gap-10 lg:grid-cols-2 lg:items-center">
            <div>
              <motion.p
                initial={{ opacity: 0, x: -20 }}
                animate={isInView ? { opacity: 1, x: 0 } : {}}
                transition={{ duration: 0.6 }}
                className="text-sm font-medium tracking-widest text-violet-400 uppercase"
              >
                The Problem
              </motion.p>

              <motion.h2
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.7, delay: 0.1 }}
                className="mt-4 text-3xl font-heading font-bold tracking-tighter text-neutral-50 md:text-5xl lg:text-6xl leading-[1.05]"
              >
                LLMs Hallucinate.{" "}
                <motion.span
                  className="text-transparent bg-clip-text bg-gradient-to-r from-red-400 to-orange-400"
                  animate={isInView ? { opacity: [0.5, 1, 0.5] } : {}}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  We Verify.
                </motion.span>
              </motion.h2>

              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.7, delay: 0.2 }}
                className="mt-6 max-w-xl text-base leading-relaxed text-neutral-300 md:text-lg"
              >
                Modern models can sound certain while being wrong. OHI turns that confidence into measurable trust.
              </motion.p>

              {/* Stats */}
              <motion.div
                className="grid grid-cols-3 gap-4 mt-8"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={isInView ? { opacity: 1, scale: 1 } : {}}
                transition={{ duration: 0.8, delay: 0.3 }}
              >
                {stats.map((stat, i) => (
                  <motion.div
                    key={stat.label}
                    className="relative flex flex-col items-center p-4 rounded-xl bg-white/5 border border-white/10"
                    initial={{ opacity: 0, y: 20 }}
                    animate={isInView ? { opacity: 1, y: 0 } : {}}
                    transition={{ delay: 0.4 + i * 0.1 }}
                    whileHover={{ scale: 1.05, borderColor: "rgba(139,92,246,0.5)" }}
                  >
                    <span className="text-2xl md:text-3xl font-bold text-white">
                      <AnimatedCounter value={stat.value} suffix={stat.suffix} />
                    </span>
                    <span className="mt-1 text-[10px] md:text-xs text-neutral-400 text-center">
                      {stat.label}
                    </span>
                  </motion.div>
                ))}
              </motion.div>
            </div>

            {/* Interactive Hallucination Visualizer */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
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
