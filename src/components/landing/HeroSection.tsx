"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { motion, type Variants } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ButtonMovingBorder } from "@/components/ui/moving-border";
import { Spotlight } from "@/components/ui/spotlight";

const KnowledgeGraphCanvas = dynamic(
  () => import("@/components/landing/_KnowledgeGraphCanvas"),
  { ssr: false }
);

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.3,
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
  return (
    <section className="relative w-full overflow-hidden">
      <div className="relative h-[44rem] w-full bg-slate-950/95 antialiased bg-grid-white/[0.03]">
        <Spotlight className="-top-40 left-0 md:left-60 md:-top-20" fill="white" />
        <motion.div
          className="absolute top-20 right-20 w-64 h-64 rounded-full bg-violet-500/30 blur-[100px]"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.4, 0.6, 0.4],
          }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="absolute bottom-40 left-10 w-96 h-96 rounded-full bg-cyan-400/20 blur-[120px]"
          animate={{
            scale: [1.2, 1, 1.2],
            opacity: [0.3, 0.5, 0.3],
          }}
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
        />
        <KnowledgeGraphCanvas />

        <motion.div
          className="relative z-10 mx-auto flex h-full max-w-7xl flex-col items-center justify-center px-4 pt-24 md:pt-0"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div
            variants={itemVariants}
            className="mb-4"
            animate={{ y: [0, -8, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
          >
            <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-neutral-200 backdrop-blur">
              <motion.span
                className="h-2 w-2 rounded-full bg-emerald-500"
                animate={{ scale: [1, 1.3, 1], opacity: [1, 0.7, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              Open Hallucination Index
              <span className="h-1 w-1 rounded-full bg-white/40" />
              Open Source
            </span>
          </motion.div>

          <motion.h1
            variants={itemVariants}
            className="text-center text-4xl font-heading font-bold tracking-tighter text-transparent md:text-7xl lg:text-8xl bg-clip-text bg-gradient-to-b from-neutral-50 via-neutral-100 to-neutral-400 leading-[0.95]"
          >
            The Trust Layer for{" "}
            <motion.span
              className="bg-gradient-to-r from-violet-400 via-cyan-400 to-emerald-400 bg-clip-text text-transparent"
              animate={{
                backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
              }}
              transition={{ duration: 5, repeat: Infinity, ease: "linear" }}
              style={{ backgroundSize: "200% 200%" }}
            >
              Artificial Intelligence.
            </motion.span>
          </motion.h1>

          <motion.p
            variants={itemVariants}
            className="mt-6 max-w-2xl text-center text-base text-neutral-300/90 md:text-lg lg:text-xl leading-relaxed font-light tracking-wide"
          >
            Decompose outputs into verifiable claims, route them through a verification oracle, and surface a trust score you can act on.
          </motion.p>

          <motion.div
            variants={itemVariants}
            className="mt-10 flex flex-col items-center gap-4 sm:flex-row"
          >
            <Link href="/auth/signup">
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.98 }}>
                <ButtonMovingBorder
                  borderRadius="1.75rem"
                  className="bg-slate-900 text-white border-slate-800"
                >
                  Get Started
                </ButtonMovingBorder>
              </motion.div>
            </Link>
            <Link href="/pricing">
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.98 }}>
                <Button
                  size="lg"
                  variant="ghost"
                  className="h-12 rounded-full border border-white/10 bg-white/5 px-8 text-neutral-200 hover:bg-white/10"
                >
                  See Pricing
                </Button>
              </motion.div>
            </Link>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
