"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { Spotlight } from "@/components/ui/spotlight";
import { ParticlesBackground } from "@/components/ui/particles-background";
import { 
  Shield, 
  Brain, 
  Network, 
  CheckCircle2, 
  Lock, 
  Zap,
  BookOpen,
  Globe,
} from "lucide-react";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.2 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } },
};

function AnimatedSection({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-50px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7 }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

const values = [
  {
    icon: Shield,
    title: "Trust Through Transparency",
    description: "Our entire methodology is openly documented. Every verification is traceable and based on verifiable sources.",
  },
  {
    icon: BookOpen,
    title: "Scientific Foundation",
    description: "Our algorithms are based on peer-reviewed research and are continuously improved through academic insights.",
  },
  {
    icon: Lock,
    title: "Privacy First",
    description: "Your data belongs to you. We don't store sensitive content and meet the highest GDPR standards.",
  },
  {
    icon: Globe,
    title: "Open Source Mission",
    description: "We believe in democratizing AI security. Our core components are freely available.",
  },
];

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-slate-950 text-neutral-100 relative overflow-hidden">
      <ParticlesBackground />
      
      {/* Hero Section */}
      <section className="relative w-full overflow-hidden">
        <div className="relative min-h-[50vh] w-full bg-slate-950/80 antialiased bg-grid-white/[0.02]">
          <Spotlight className="-top-40 left-0 md:left-60 md:-top-20" fill="white" />
          
          <motion.div
            className="absolute top-10 right-10 md:top-20 md:right-20 w-48 md:w-64 h-48 md:h-64 rounded-full bg-violet-500/30 blur-[80px]"
            animate={{ scale: [1, 1.2, 1], opacity: [0.4, 0.6, 0.4] }}
            transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          />
          <motion.div
            className="absolute bottom-10 left-5 md:bottom-20 md:left-10 w-64 md:w-96 h-64 md:h-96 rounded-full bg-cyan-400/25 blur-[100px]"
            animate={{ scale: [1.2, 1, 1.2], opacity: [0.3, 0.5, 0.3] }}
            transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
          />

          <motion.div
            className="relative z-10 mx-auto flex min-h-[50vh] max-w-5xl flex-col items-center justify-center px-4 py-16 text-center"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            <motion.div variants={itemVariants} className="mb-6">
              <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-neutral-200 backdrop-blur">
                <motion.span
                  className="h-2 w-2 rounded-full bg-emerald-500"
                  animate={{ scale: [1, 1.3, 1], opacity: [1, 0.7, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
                About Us
              </span>
            </motion.div>

            <motion.h1
              variants={itemVariants}
              className="text-4xl md:text-6xl lg:text-7xl font-heading font-bold tracking-tighter"
            >
              The Future of{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 via-cyan-400 to-emerald-400">
                AI Verification
              </span>
            </motion.h1>

            <motion.p
              variants={itemVariants}
              className="mt-6 max-w-3xl text-lg md:text-xl text-neutral-300 leading-relaxed"
            >
              The Open Hallucination Index is the first independent, open-source platform 
              for detecting and verifying AI hallucinations. We build trust 
              in a world where AI-generated content is becoming ubiquitous.
            </motion.p>
          </motion.div>
        </div>
      </section>

      {/* Mission Statement */}
      <AnimatedSection className="relative py-20 md:py-28">
        <div className="mx-auto max-w-7xl px-4">
          <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/80 via-slate-900/60 to-slate-800/40 p-8 md:p-14 backdrop-blur-xl">
            <div
              className="absolute inset-0 rounded-3xl opacity-70 animate-shimmer"
              style={{
                background: "linear-gradient(90deg, transparent 0%, rgba(100,80,180,0.5) 25%, rgba(167,139,250,0.6) 50%, rgba(100,80,180,0.5) 75%, transparent 100%)",
                backgroundSize: "200% 100%",
              }}
            />
            
            <div className="relative z-10 grid gap-10 lg:grid-cols-2 lg:items-center">
              <div>
                <p className="text-sm font-medium tracking-widest text-violet-400 uppercase">
                  Our Mission
                </p>
                <h2 className="mt-4 text-3xl md:text-4xl lg:text-5xl font-heading font-bold tracking-tight text-neutral-50 leading-tight">
                  Truth as an{" "}
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-violet-400">
                    API Endpoint
                  </span>
                </h2>
                <p className="mt-6 text-neutral-300 leading-relaxed text-lg">
                  Large Language Models have revolutionized how we interact with information. 
                  But with great power comes great responsibility: Up to 27% of all LLM outputs contain 
                  factual errors – so-called hallucinations.
                </p>
                <p className="mt-4 text-neutral-400 leading-relaxed">
                  We have made it our mission to close this trust gap. Not through 
                  censorship or restriction, but through transparent, traceable verification.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                {[
                  { icon: Brain, label: "Claim Decomposition", value: "Atomic Analysis" },
                  { icon: Network, label: "Knowledge Graph", value: "Hybrid Verification" },
                  { icon: CheckCircle2, label: "Citation Trace", value: "Traceable" },
                  { icon: Zap, label: "Real-time", value: "<50ms Latency" },
                ].map((item, i) => (
                  <motion.div
                    key={item.label}
                    initial={{ opacity: 0, scale: 0.9 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.1, duration: 0.5 }}
                    viewport={{ once: true }}
                    className="rounded-2xl border border-white/10 bg-slate-800/50 p-5 backdrop-blur-sm"
                  >
                    <item.icon className="h-8 w-8 text-violet-400 mb-3" />
                    <p className="text-sm text-neutral-400">{item.label}</p>
                    <p className="text-lg font-semibold text-neutral-100">{item.value}</p>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </AnimatedSection>

      {/* How It Works */}
      <AnimatedSection className="py-20 md:py-28 bg-slate-900/30">
        <div className="mx-auto max-w-7xl px-4">
          <div className="text-center mb-16">
            <p className="text-sm font-medium tracking-widest text-cyan-400 uppercase">
              Technology
            </p>
            <h2 className="mt-4 text-3xl md:text-4xl lg:text-5xl font-heading font-bold tracking-tight text-neutral-50">
              How Verification Works
            </h2>
            <p className="mt-4 max-w-2xl mx-auto text-neutral-400">
              A multi-step process that combines linguistic analysis with knowledge graph technology.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-3">
            {[
              {
                step: "01",
                title: "Claim Decomposition",
                description: "The Input Processor breaks down complex texts into atomic subject-predicate-object triplets. Each individual claim is isolated and analyzed independently.",
                gradient: "from-violet-500 to-purple-600",
              },
              {
                step: "02",
                title: "Knowledge Graph Matching",
                description: "The Verification Oracle matches triplets against a hybrid index: Trusted domains (government data, science) combined with semantic consensus graphs.",
                gradient: "from-cyan-500 to-blue-600",
              },
              {
                step: "03",
                title: "Scoring & Citation",
                description: "Each claim receives a HallucinationScore (0.0–1.0) and a complete Citation Trace with direct links to confirming or refuting sources.",
                gradient: "from-emerald-500 to-teal-600",
              },
            ].map((item, i) => (
              <motion.div
                key={item.step}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.15, duration: 0.6 }}
                viewport={{ once: true }}
                className="relative group"
              >
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-br opacity-0 group-hover:opacity-100 transition-opacity duration-500 -z-10 blur-xl"
                  style={{ background: `linear-gradient(135deg, var(--tw-gradient-stops))` }}
                />
                <div className="h-full rounded-2xl border border-white/10 bg-slate-900/60 p-8 backdrop-blur-sm transition-all duration-300 group-hover:border-white/20 group-hover:bg-slate-900/80">
                  <span className={`inline-block text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r ${item.gradient} opacity-50`}>
                    {item.step}
                  </span>
                  <h3 className="mt-4 text-xl font-semibold text-neutral-100">{item.title}</h3>
                  <p className="mt-3 text-neutral-400 leading-relaxed">{item.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </AnimatedSection>

      {/* Values */}
      <AnimatedSection className="py-20 md:py-28">
        <div className="mx-auto max-w-7xl px-4">
          <div className="text-center mb-16">
            <p className="text-sm font-medium tracking-widest text-emerald-400 uppercase">
              Our Values
            </p>
            <h2 className="mt-4 text-3xl md:text-4xl lg:text-5xl font-heading font-bold tracking-tight text-neutral-50">
              Principles That Guide Us
            </h2>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {values.map((value, i) => (
              <motion.div
                key={value.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1, duration: 0.5 }}
                viewport={{ once: true }}
                className="rounded-2xl border border-white/10 bg-slate-900/40 p-6 backdrop-blur-sm hover:border-white/20 transition-colors"
              >
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500/20 to-cyan-500/20 flex items-center justify-center mb-4">
                  <value.icon className="h-6 w-6 text-violet-400" />
                </div>
                <h3 className="text-lg font-semibold text-neutral-100">{value.title}</h3>
                <p className="mt-2 text-sm text-neutral-400 leading-relaxed">{value.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </AnimatedSection>

      {/* CTA Section */}
      <AnimatedSection className="py-20 md:py-28">
        <div className="mx-auto max-w-4xl px-4 text-center">
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-heading font-bold tracking-tight text-neutral-50">
            Ready for Verified AI?
          </h2>
          <p className="mt-6 text-lg text-neutral-300 max-w-2xl mx-auto">
            Get started with the Open Hallucination Index today and bring 
            trust to your AI applications.
          </p>
          <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center">
            <motion.a
              href="/pricing"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="inline-flex items-center justify-center px-8 py-4 rounded-xl bg-gradient-to-r from-violet-600 to-cyan-600 text-white font-semibold text-lg shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 transition-shadow"
            >
              Start for Free
            </motion.a>
            <motion.a
              href="https://github.com/open-hallucination-index"
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="inline-flex items-center justify-center px-8 py-4 rounded-xl border border-white/20 bg-white/5 text-neutral-100 font-semibold text-lg hover:bg-white/10 transition-colors"
            >
              View on GitHub
            </motion.a>
          </div>
        </div>
      </AnimatedSection>

      {/* Footer spacing */}
      <div className="h-10" />
    </main>
  );
}
