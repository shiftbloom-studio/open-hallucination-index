"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const features = [
  {
    title: "GraphRAG",
    description: "Beyond simple vector search—model claims against a structured graph of entities, relations, and citations.",
    icon: "🔗",
    gradient: "from-violet-400/30 to-purple-500/30",
    colSpan: "lg:col-span-3",
  },
  {
    title: "Atomic Verification",
    description: "Checking facts, not just tokens. Each statement is scored independently so failures are visible and actionable.",
    icon: "⚛️",
    gradient: "from-cyan-400/30 to-blue-500/30",
    colSpan: "lg:col-span-3",
  },
  {
    title: "Open Source",
    description: "Transparent and community-driven—inspect the methodology, reproduce results, and contribute improvements.",
    icon: "🌐",
    gradient: "from-emerald-400/30 to-teal-500/30",
    colSpan: "lg:col-span-6",
  },
];

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
          <span className="text-xs font-medium tracking-widest text-violet-400 uppercase">
            Features
          </span>
          <h2 className="mt-3 text-3xl font-heading font-bold tracking-tighter text-neutral-50 md:text-5xl lg:text-6xl leading-[1.05]">
            Built for grounded systems.
          </h2>
          <p className="mt-4 text-base leading-relaxed text-neutral-300/90 md:text-lg lg:text-xl font-light tracking-wide">
            A verification-first stack for teams that need reliable outputs.
          </p>
        </motion.div>

        <div className="mt-10 grid gap-4 lg:grid-cols-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              className={feature.colSpan}
              initial={{ opacity: 0, y: 40, scale: 0.95 }}
              animate={isInView ? { opacity: 1, y: 0, scale: 1 } : {}}
              transition={{ duration: 0.6, delay: 0.2 + index * 0.15 }}
            >
              <motion.div
                whileHover={{
                  scale: 1.02,
                  y: -4,
                  transition: { duration: 0.2 },
                }}
                className="h-full"
              >
                <Card className={cn(
                  "h-full border-white/15 bg-white/[0.08] backdrop-blur-xl overflow-hidden relative group cursor-pointer transition-all duration-300 hover:border-white/30 hover:bg-white/[0.12]"
                )}>
                  {/* Hover gradient */}
                  <motion.div
                    className={cn(
                      "absolute inset-0 bg-gradient-to-br opacity-0 group-hover:opacity-100 transition-opacity duration-500",
                      feature.gradient
                    )}
                  />
                  
                  {/* Shimmer effect on hover */}
                  <motion.div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100"
                    style={{
                      background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent)",
                      backgroundSize: "200% 100%",
                    }}
                    animate={{ backgroundPosition: ["-100% 0%", "200% 0%"] }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                  />

                  <CardHeader className="relative z-10">
                    <div className="flex items-center gap-3">
                      <motion.span
                        className="text-2xl"
                        animate={isInView ? { rotate: [0, 10, -10, 0] } : {}}
                        transition={{ duration: 0.5, delay: 0.5 + index * 0.1 }}
                      >
                        {feature.icon}
                      </motion.span>
                      <CardTitle className="text-neutral-50">{feature.title}</CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent className="relative z-10 text-neutral-300">
                    {feature.description}
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
