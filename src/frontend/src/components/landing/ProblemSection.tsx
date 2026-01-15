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
    <span ref={ref} className="tabular-nums">
      {displayValue}{suffix}
    </span>
  );
}

// Hallucinated vs Verified claims for the animation
// wrongWord marks the incorrect part in the hallucinated text (only for false claims)
const claims = [
  { 
    hallucinated: "The Eiffel Tower was built in 1920",
    verified: "The Eiffel Tower was built in 1889",
    wrongWord: "1920",
    isTrue: false 
  },
  { 
    hallucinated: "Water boils at 100°C at sea level",
    verified: "Water boils at 100°C at sea level",
    wrongWord: null,
    isTrue: true 
  },
  { 
    hallucinated: "Shakespeare wrote 47 plays",
    verified: "Shakespeare wrote ~37 plays",
    wrongWord: "47",
    isTrue: false 
  },
  { 
    hallucinated: "The human body has 206 bones",
    verified: "The human body has 206 bones",
    wrongWord: null,
    isTrue: true 
  },
  { 
    hallucinated: "Einstein discovered gravity in 1687",
    verified: "Newton discovered gravity in 1687",
    wrongWord: "Einstein",
    isTrue: false 
  },
  { 
    hallucinated: "Mars has 3 moons orbiting it",
    verified: "Mars has 2 moons orbiting it",
    wrongWord: "3",
    isTrue: false 
  },
  { 
    hallucinated: "The speed of light is ~300,000 km/s",
    verified: "The speed of light is ~300,000 km/s",
    wrongWord: null,
    isTrue: true 
  },
  { 
    hallucinated: "The Amazon is 8,000 km long",
    verified: "The Amazon is ~6,400 km long",
    wrongWord: "8,000",
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
          .map((char) => {
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

// Component to highlight the wrong word in a claim
function HighlightedClaimText({ 
  text, 
  wrongWord, 
  showHighlight 
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
    // wrongWord not found in text
    return <span>{text}</span>;
  }
  
  return (
    <span>
      {parts[0]}
      <motion.span
        className="relative inline-block"
        initial={{ backgroundColor: "transparent" }}
        animate={{ 
          backgroundColor: "rgba(239, 68, 68, 0.3)",
        }}
        transition={{ duration: 0.3 }}
      >
        <span className="text-red-400 font-semibold relative">
          {wrongWord}
          <motion.span
            className="absolute -bottom-0.5 left-0 right-0 h-0.5 bg-red-500"
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

// Deterministic positions for particles to avoid hydration mismatch
const PARTICLE_POSITIONS = [12, 28, 45, 62, 78, 88, 35, 55];

// Floating verification particles
function VerificationParticle({ delay, verified, index }: { delay: number; verified: boolean; index: number }) {
  return (
    <motion.div
      className={`absolute w-1.5 h-1.5 rounded-full ${verified ? "bg-emerald-400" : "bg-red-400"}`}
      style={{
        left: `${PARTICLE_POSITIONS[index % PARTICLE_POSITIONS.length]}%`,
      }}
      initial={{ top: "50%", opacity: 0, scale: 0 }}
      animate={{
        top: verified ? "-20%" : "120%",
        opacity: [0, 1, 1, 0],
        scale: [0, 1, 1, 0],
        boxShadow: verified 
          ? ["0 0 8px rgba(52,211,153,0.8)", "0 0 12px rgba(52,211,153,0.9)", "0 0 8px rgba(52,211,153,0.8)"]
          : ["0 0 8px rgba(248,113,113,0.8)", "0 0 12px rgba(248,113,113,0.9)", "0 0 8px rgba(248,113,113,0.8)"],
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

// Animation state type for synchronized state management
// Phases: analyzing -> scanning -> detected (only for false claims) -> verified -> (wait) -> next claim
type AnimationPhase = "analyzing" | "scanning" | "detected" | "verified";

interface AnimationState {
  claimIndex: number;
  phase: AnimationPhase;
}

// Custom hook for synchronized animation state management
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
    
    // Phase durations in ms
    const PHASE_DURATIONS = {
      analyzing: 1500,
      scanning: 1800,
      detected: 1500,  // Only used for false claims
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
            // For true claims, skip detected phase and go directly to verified
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
        // Atomic state update - both phase and claim index update together
        setState({ phase: nextPhase, claimIndex: nextClaimIndex });
        scheduleNextPhase(nextPhase, nextClaimIndex);
      }, delay);
    };
    
    // Start with analyzing phase
    setState({ phase: "analyzing", claimIndex: 0 });
    scheduleNextPhase("analyzing", 0);
    
    // Copy the ref value to a stable variable for the cleanup function
    const currentAnimation = animationRef.current;
    
    return () => {
      if (currentAnimation.timeoutId) {
        clearTimeout(currentAnimation.timeoutId);
      }
    };
  }, [isInView]);
  
  return state;
}

// Main hallucination visualization component
function HallucinationVisualizer({ isInView }: { isInView: boolean }) {
  // Use synchronized state manager
  const { claimIndex, phase } = useAnimationState(isInView);
  
  // Derived state - claim is always in sync with the animation state
  const claim = claims[claimIndex];
  
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
          index={i}
        />
      ))}
      
      {/* Neural connections */}
      <NeuralConnection startX={20} startY={180} endX={180} endY={40} delay={0} />
      <NeuralConnection startX={280} startY={40} endX={380} endY={180} delay={0.5} />
      
      {/* Scanner beam */}
      <ScannerBeam isActive={phase === "scanning"} />
      
      {/* Main content area */}
      <div className="relative z-10 flex flex-col items-center justify-center h-full px-2 sm:px-4">
        {/* Status indicator */}
        <motion.div 
          className="flex items-center gap-2 mb-3 sm:mb-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <motion.div
            className={`w-2 h-2 rounded-full ${
              phase === "analyzing" ? "bg-violet-500" :
              phase === "scanning" ? "bg-cyan-400" :
              phase === "detected" ? "bg-red-500" :
              claim.isTrue ? "bg-emerald-500" : "bg-emerald-500"
            }`}
            animate={{ 
              scale: phase === "scanning" ? [1, 1.5, 1] : 1,
              boxShadow: phase === "scanning" 
                ? ["0 0 0px rgba(34,211,238,0.5)", "0 0 20px rgba(34,211,238,0.8)", "0 0 0px rgba(34,211,238,0.5)"]
                : "none"
            }}
            transition={{ duration: 0.5, repeat: phase === "scanning" ? Infinity : 0 }}
          />
          <span className={`text-[10px] sm:text-xs font-medium uppercase tracking-wider ${
            phase === "analyzing" ? "text-violet-400" :
            phase === "scanning" ? "text-cyan-400" :
            phase === "detected" ? "text-red-400" :
            "text-emerald-400"
          }`}>
            {phase === "analyzing" ? "Analyzing Claim..." :
             phase === "scanning" ? "Verifying..." :
             phase === "detected" ? "Hallucination Detected!" :
             claim.isTrue ? "Verified ✓" : "Corrected ✓"}
          </span>
        </motion.div>
        
        {/* Claim text box */}
        <motion.div
          className={`relative px-3 sm:px-6 py-3 sm:py-4 rounded-lg border backdrop-blur-sm transition-all duration-500 max-w-full ${
            phase === "analyzing" 
              ? "bg-violet-500/10 border-violet-500/30" 
              : phase === "scanning"
              ? "bg-cyan-500/10 border-cyan-500/30"
              : phase === "detected"
              ? "bg-red-500/10 border-red-500/30"
              : "bg-emerald-500/10 border-emerald-500/30"
          }`}
          animate={{
            x: phase === "detected" ? [0, -2, 2, -1, 1, 0] : 0,
          }}
          transition={{ 
            duration: 0.3, 
            repeat: phase === "detected" ? 3 : 0,
            repeatDelay: 0.1,
          }}
        >
          {/* Glitch lines overlay for detected phase */}
          {phase === "detected" && (
            <>
              <motion.div
                className="absolute inset-0 bg-red-500/10 rounded-lg"
                animate={{ opacity: [0, 0.4, 0] }}
                transition={{ duration: 0.2, repeat: 3 }}
              />
            </>
          )}
          
          <p className="text-xs sm:text-sm md:text-base text-white/90 text-center font-mono leading-relaxed">
            <AnimatePresence mode="wait">
              {phase === "verified" ? (
                <motion.span
                  key="verified"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="text-emerald-300"
                >
                  {claim.verified}
                </motion.span>
              ) : phase === "detected" ? (
                <motion.span
                  key="detected"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <HighlightedClaimText 
                    text={claim.hallucinated} 
                    wrongWord={claim.wrongWord} 
                    showHighlight={true}
                  />
                </motion.span>
              ) : (
                <motion.span
                  key="analyzing"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <GlitchText 
                    text={claim.hallucinated} 
                    isGlitching={phase === "analyzing"} 
                  />
                </motion.span>
              )}
            </AnimatePresence>
          </p>
        </motion.div>
        
        {/* Trust score meter */}
        <motion.div 
          className="mt-4 sm:mt-6 w-full max-w-[180px] sm:max-w-[200px]"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex justify-between text-[9px] sm:text-[10px] text-white/50 mb-1">
            <span>0%</span>
            <span className="text-white/70 font-medium">Trust Score</span>
            <span>100%</span>
          </div>
          <div className="h-1 sm:h-1.5 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              className={`h-full rounded-full ${
                phase === "verified"
                  ? "bg-gradient-to-r from-emerald-500 to-emerald-400"
                  : phase === "detected"
                  ? "bg-gradient-to-r from-red-600 to-red-500"
                  : "bg-gradient-to-r from-violet-500 to-cyan-400"
              }`}
              initial={{ width: "0%" }}
              animate={{ 
                width: phase === "analyzing" ? "50%" :
                       phase === "scanning" ? "50%" :
                       phase === "detected" ? "5%" :
                       "100%"
              }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            />
          </div>
        </motion.div>
        
        {/* Claim counter */}
        <div className="absolute bottom-2 right-2 sm:right-3 flex gap-0.5 sm:gap-1">
          {claims.map((_, i) => (
            <motion.div
              key={i}
              className={`w-1 h-1 rounded-full ${
                i === claimIndex ? "bg-violet-400" : "bg-white/20"
              }`}
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
          className="relative overflow-hidden rounded-3xl border border-white/15 bg-slate-900/60 px-6 py-12 backdrop-blur-xl md:px-14"
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          {/* Animated gradient border */}
          <motion.div
            className="absolute inset-0 rounded-3xl opacity-60"
            style={{
              // Start and end colors are identical to ensure a seamless loop
              background: "linear-gradient(90deg, rgba(167,139,250,0.4) 0%, rgba(34,211,238,0.4) 50%, rgba(167,139,250,0.4) 100%)",
              backgroundSize: "200% 100%",
            }}
            // animate backgroundPosition from 0% -> 100% for a smooth translation
            animate={{ backgroundPosition: ["0% 0%", "100% 0%"] }}
            // slowed down for a calmer effect; adjust duration as needed
            transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
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
                {stats.map((stat, index) => (
                  <motion.div
                    key={stat.label}
                    className="relative flex flex-col items-center p-4 rounded-xl bg-white/5 border border-white/10"
                    initial={{ opacity: 0, y: 20 }}
                    animate={isInView ? { opacity: 1, y: 0 } : {}}
                    transition={{ delay: 0.4 + index * 0.1 }}
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
