"use client";

import Link from "next/link";
import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { Button } from "@/components/ui/button";
import { ButtonMovingBorder } from "@/components/ui/moving-border";

function FloatingOrb({ className, delay = 0 }: { className: string; delay?: number }) {
  return (
    <motion.div
      className={className}
      animate={{
        y: [-20, 20, -20],
        x: [-10, 10, -10],
        scale: [1, 1.1, 1],
      }}
      transition={{
        duration: 8,
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
          className="relative overflow-hidden rounded-3xl border border-white/15 bg-slate-900/60 p-10 backdrop-blur-xl md:p-16"
          initial={{ opacity: 0, y: 50 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          {/* Floating orbs */}
          <FloatingOrb
            className="absolute -top-20 -left-20 w-40 h-40 rounded-full bg-violet-500/40 blur-[80px]"
            delay={0}
          />
          <FloatingOrb
            className="absolute -bottom-20 -right-20 w-56 h-56 rounded-full bg-cyan-400/30 blur-[100px]"
            delay={2}
          />
          <FloatingOrb
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-72 h-72 rounded-full bg-emerald-400/20 blur-[120px]"
            delay={4}
          />

          {/* Animated border gradient */}
          <motion.div
            className="absolute inset-0 rounded-3xl"
            style={{
              background: "linear-gradient(90deg, transparent, rgba(167,139,250,0.12), rgba(34,211,238,0.12), transparent)",
              backgroundSize: "300% 100%",
            }}
            animate={{ backgroundPosition: ["-100% 0%", "200% 0%"] }}
            transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
          />

          <div className="relative z-10 flex flex-col items-center text-center">
            <motion.h2
              className="text-3xl font-heading font-bold tracking-tighter text-neutral-50 md:text-5xl lg:text-6xl leading-[1.05]"
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.2 }}
            >
              Make hallucinations{" "}
              <motion.span
                className="relative inline-block text-transparent bg-clip-text bg-gradient-to-r from-violet-400 via-cyan-400 to-emerald-400"
                initial={{ opacity: 0, filter: "blur(8px)" }}
                animate={isInView ? { 
                  opacity: 1, 
                  filter: "blur(0px)",
                } : {}}
                transition={{ 
                  duration: 1.2, 
                  delay: 0.5,
                  ease: [0.25, 0.46, 0.45, 0.94]
                }}
              >
                measurable.
                <motion.span
                  className="absolute -inset-1 bg-gradient-to-r from-violet-400/20 via-cyan-400/20 to-emerald-400/20 blur-xl rounded-lg -z-10"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={isInView ? { 
                    opacity: [0, 0.6, 0.3],
                    scale: [0.8, 1.1, 1]
                  } : {}}
                  transition={{ 
                    duration: 2,
                    delay: 0.6,
                    ease: "easeOut"
                  }}
                />
              </motion.span>
            </motion.h2>

            <motion.p
              className="mt-6 max-w-2xl text-base leading-relaxed text-neutral-300/90 md:text-lg lg:text-xl font-light tracking-wide"
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.3 }}
            >
              Start verifying today—run it locally, integrate it into your pipeline, and ship AI you can trust.
            </motion.p>

            <motion.div
              className="mt-10 flex flex-col items-center gap-4 sm:flex-row"
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.4 }}
            >
              <Link href="/auth/signup">
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.98 }}>
                  <ButtonMovingBorder
                    borderRadius="1.75rem"
                    className="bg-slate-900 text-white border-slate-800 h-12 px-8"
                  >
                    Get Started
                  </ButtonMovingBorder>
                </motion.div>
              </Link>

              <Link href="/about">
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.98 }}>
                  <Button
                    size="lg"
                    variant="ghost"
                    className="h-12 rounded-full border border-white/10 bg-white/5 px-8 text-neutral-200 hover:bg-white/10"
                  >
                    Learn More
                  </Button>
                </motion.div>
              </Link>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
