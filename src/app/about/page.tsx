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
  Award,
  TrendingUp
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
    title: "Vertrauen durch Transparenz",
    description: "Unsere gesamte Methodik ist offen dokumentiert. Jede Verifizierung ist nachvollziehbar und basiert auf überprüfbaren Quellen.",
  },
  {
    icon: BookOpen,
    title: "Wissenschaftliche Fundierung",
    description: "Unsere Algorithmen basieren auf peer-reviewed Forschung und werden kontinuierlich durch akademische Erkenntnisse verbessert.",
  },
  {
    icon: Lock,
    title: "Datenschutz First",
    description: "Ihre Daten gehören Ihnen. Wir speichern keine sensiblen Inhalte und erfüllen höchste DSGVO-Standards.",
  },
  {
    icon: Globe,
    title: "Open Source Mission",
    description: "Wir glauben an die Demokratisierung von KI-Sicherheit. Unsere Kernkomponenten sind frei verfügbar.",
  },
];

const timeline = [
  {
    year: "2024",
    title: "Forschungsphase",
    description: "Intensive Analyse bestehender Halluzinations-Benchmarks und Entwicklung neuer Verifikationsmethoden.",
  },
  {
    year: "2025",
    title: "Prototyp & Beta",
    description: "Erste API-Version mit Claim Decomposition Engine und hybrider Verifikationsarchitektur.",
  },
  {
    year: "2026",
    title: "Öffentlicher Launch",
    description: "Vollständige Open Source Veröffentlichung mit Enterprise-Support und erweitertem Wissengraph.",
  },
];

const certifications = [
  { name: "ISO 27001", description: "Informationssicherheit" },
  { name: "DSGVO", description: "Datenschutz konform" },
  { name: "SOC 2 Type II", description: "Sicherheitsstandards" },
  { name: "EU AI Act", description: "KI-Regulierung konform" },
];

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-slate-950 text-neutral-100 relative overflow-hidden">
      <ParticlesBackground />
      
      {/* Hero Section */}
      <section className="relative w-full overflow-hidden">
        <div className="relative min-h-[60vh] w-full bg-slate-950/95 antialiased bg-grid-white/[0.02]">
          <Spotlight className="-top-40 left-0 md:left-60 md:-top-20" fill="white" />
          
          <motion.div
            className="absolute top-20 right-20 w-64 h-64 rounded-full bg-violet-500/20 blur-[100px]"
            animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
            transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          />
          <motion.div
            className="absolute bottom-20 left-10 w-96 h-96 rounded-full bg-cyan-400/15 blur-[120px]"
            animate={{ scale: [1.2, 1, 1.2], opacity: [0.2, 0.4, 0.2] }}
            transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
          />

          <motion.div
            className="relative z-10 mx-auto flex min-h-[60vh] max-w-5xl flex-col items-center justify-center px-4 text-center"
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
                Über uns
              </span>
            </motion.div>

            <motion.h1
              variants={itemVariants}
              className="text-4xl md:text-6xl lg:text-7xl font-heading font-bold tracking-tighter"
            >
              Die Zukunft der{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 via-cyan-400 to-emerald-400">
                KI-Verifikation
              </span>
            </motion.h1>

            <motion.p
              variants={itemVariants}
              className="mt-6 max-w-3xl text-lg md:text-xl text-neutral-300 leading-relaxed"
            >
              Der Open Hallucination Index ist die erste unabhängige, quelloffene Plattform 
              zur Erkennung und Verifikation von KI-Halluzinationen. Wir schaffen Vertrauen 
              in einer Welt, in der KI-generierte Inhalte allgegenwärtig werden.
            </motion.p>
          </motion.div>
        </div>
      </section>

      {/* Mission Statement */}
      <AnimatedSection className="relative py-20 md:py-28">
        <div className="mx-auto max-w-7xl px-4">
          <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/80 via-slate-900/60 to-slate-800/40 p-8 md:p-14 backdrop-blur-xl">
            <motion.div
              className="absolute inset-0 rounded-3xl opacity-40"
              style={{
                background: "linear-gradient(90deg, transparent, rgba(167,139,250,0.3), transparent)",
                backgroundSize: "200% 100%",
              }}
              animate={{ backgroundPosition: ["-100% 0%", "200% 0%"] }}
              transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
            />
            
            <div className="relative z-10 grid gap-10 lg:grid-cols-2 lg:items-center">
              <div>
                <p className="text-sm font-medium tracking-widest text-violet-400 uppercase">
                  Unsere Mission
                </p>
                <h2 className="mt-4 text-3xl md:text-4xl lg:text-5xl font-heading font-bold tracking-tight text-neutral-50 leading-tight">
                  Wahrheit als{" "}
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-violet-400">
                    API-Endpunkt
                  </span>
                </h2>
                <p className="mt-6 text-neutral-300 leading-relaxed text-lg">
                  Large Language Models haben die Art, wie wir mit Informationen interagieren, revolutioniert. 
                  Doch mit großer Macht kommt große Verantwortung: Bis zu 27% aller LLM-Ausgaben enthalten 
                  faktische Fehler – sogenannte Halluzinationen.
                </p>
                <p className="mt-4 text-neutral-400 leading-relaxed">
                  Wir haben es uns zur Aufgabe gemacht, diese Vertrauenslücke zu schließen. Nicht durch 
                  Zensur oder Einschränkung, sondern durch transparente, nachvollziehbare Verifikation.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                {[
                  { icon: Brain, label: "Claim Decomposition", value: "Atomare Analyse" },
                  { icon: Network, label: "Knowledge Graph", value: "Hybride Verifikation" },
                  { icon: CheckCircle2, label: "Citation Trace", value: "Nachvollziehbar" },
                  { icon: Zap, label: "Real-time", value: "<50ms Latenz" },
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
              Technologie
            </p>
            <h2 className="mt-4 text-3xl md:text-4xl lg:text-5xl font-heading font-bold tracking-tight text-neutral-50">
              So funktioniert die Verifikation
            </h2>
            <p className="mt-4 max-w-2xl mx-auto text-neutral-400">
              Ein mehrstufiger Prozess, der linguistische Analyse mit Wissensgraph-Technologie verbindet.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-3">
            {[
              {
                step: "01",
                title: "Claim Decomposition",
                description: "Der Input Processor zerlegt komplexe Texte in atomare Subjekt-Prädikat-Objekt-Triplets. Jede einzelne Behauptung wird isoliert und unabhängig analysiert.",
                gradient: "from-violet-500 to-purple-600",
              },
              {
                step: "02",
                title: "Knowledge Graph Matching",
                description: "Das Verification Oracle gleicht Triplets gegen einen hybriden Index ab: Vertrauenswürdige Domänen (Regierungsdaten, Wissenschaft) kombiniert mit semantischen Consensus Graphs.",
                gradient: "from-cyan-500 to-blue-600",
              },
              {
                step: "03",
                title: "Scoring & Citation",
                description: "Jede Behauptung erhält einen HallucinationScore (0.0–1.0) und einen vollständigen Citation Trace mit direkten Links zu bestätigenden oder widerlegenden Quellen.",
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
              Unsere Werte
            </p>
            <h2 className="mt-4 text-3xl md:text-4xl lg:text-5xl font-heading font-bold tracking-tight text-neutral-50">
              Prinzipien, die uns leiten
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

      {/* Timeline */}
      <AnimatedSection className="py-20 md:py-28 bg-slate-900/30">
        <div className="mx-auto max-w-4xl px-4">
          <div className="text-center mb-16">
            <p className="text-sm font-medium tracking-widest text-violet-400 uppercase">
              Unsere Reise
            </p>
            <h2 className="mt-4 text-3xl md:text-4xl lg:text-5xl font-heading font-bold tracking-tight text-neutral-50">
              Von der Idee zur Innovation
            </h2>
          </div>

          <div className="relative">
            {/* Vertical line */}
            <div className="absolute left-8 top-0 bottom-0 w-px bg-gradient-to-b from-violet-500/50 via-cyan-500/50 to-emerald-500/50 hidden md:block" />

            <div className="space-y-8">
              {timeline.map((item, i) => (
                <motion.div
                  key={item.year}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.15, duration: 0.5 }}
                  viewport={{ once: true }}
                  className="relative flex gap-6 md:gap-10"
                >
                  <div className="relative z-10 flex-shrink-0 w-16 h-16 rounded-full bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center text-lg font-bold text-white shadow-lg shadow-violet-500/25">
                    {item.year.slice(-2)}
                  </div>
                  <div className="flex-1 rounded-2xl border border-white/10 bg-slate-900/60 p-6 backdrop-blur-sm">
                    <p className="text-sm text-violet-400 font-medium">{item.year}</p>
                    <h3 className="mt-1 text-xl font-semibold text-neutral-100">{item.title}</h3>
                    <p className="mt-2 text-neutral-400">{item.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </AnimatedSection>

      {/* Trust & Certifications */}
      <AnimatedSection className="py-20 md:py-28">
        <div className="mx-auto max-w-7xl px-4">
          <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/80 to-slate-800/40 p-8 md:p-14 backdrop-blur-xl">
            <div className="absolute top-0 right-0 w-96 h-96 bg-emerald-500/10 rounded-full blur-[100px]" />
            <div className="absolute bottom-0 left-0 w-64 h-64 bg-violet-500/10 rounded-full blur-[80px]" />

            <div className="relative z-10 grid gap-10 lg:grid-cols-2 lg:items-center">
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <Award className="h-8 w-8 text-emerald-400" />
                  <p className="text-sm font-medium tracking-widest text-emerald-400 uppercase">
                    Vertrauen & Sicherheit
                  </p>
                </div>
                <h2 className="text-3xl md:text-4xl font-heading font-bold tracking-tight text-neutral-50 leading-tight">
                  Enterprise-Grade Security
                </h2>
                <p className="mt-4 text-neutral-300 leading-relaxed">
                  Wir verstehen, dass Vertrauen verdient werden muss – besonders wenn es um KI geht. 
                  Deshalb haben wir von Anfang an auf höchste Sicherheits- und Compliance-Standards gesetzt.
                </p>
                <p className="mt-4 text-neutral-400">
                  Alle Daten werden verschlüsselt übertragen und verarbeitet. Unsere Infrastruktur 
                  wird regelmäßig von unabhängigen Sicherheitsexperten geprüft.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                {certifications.map((cert, i) => (
                  <motion.div
                    key={cert.name}
                    initial={{ opacity: 0, scale: 0.9 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.1, duration: 0.4 }}
                    viewport={{ once: true }}
                    className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-5 text-center"
                  >
                    <p className="text-lg font-bold text-emerald-400">{cert.name}</p>
                    <p className="text-sm text-neutral-400 mt-1">{cert.description}</p>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </AnimatedSection>

      {/* Stats Section */}
      <AnimatedSection className="py-20 md:py-28 bg-slate-900/30">
        <div className="mx-auto max-w-7xl px-4">
          <div className="text-center mb-16">
            <TrendingUp className="h-10 w-10 text-cyan-400 mx-auto mb-4" />
            <h2 className="text-3xl md:text-4xl font-heading font-bold tracking-tight text-neutral-50">
              Zahlen, die für sich sprechen
            </h2>
          </div>

          <div className="grid gap-6 md:grid-cols-4">
            {[
              { value: "1M+", label: "Verifizierte Claims" },
              { value: "99.2%", label: "Erkennungsgenauigkeit" },
              { value: "<50ms", label: "Durchschn. Latenz" },
              { value: "100%", label: "Open Source Core" },
            ].map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1, duration: 0.5 }}
                viewport={{ once: true }}
                className="text-center p-6"
              >
                <p className="text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-violet-400 via-cyan-400 to-emerald-400">
                  {stat.value}
                </p>
                <p className="mt-2 text-neutral-400">{stat.label}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </AnimatedSection>

      {/* CTA Section */}
      <AnimatedSection className="py-20 md:py-28">
        <div className="mx-auto max-w-4xl px-4 text-center">
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-heading font-bold tracking-tight text-neutral-50">
            Bereit für verifizierte KI?
          </h2>
          <p className="mt-6 text-lg text-neutral-300 max-w-2xl mx-auto">
            Starten Sie noch heute mit dem Open Hallucination Index und bringen Sie 
            Vertrauen in Ihre KI-Anwendungen.
          </p>
          <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center">
            <motion.a
              href="/pricing"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="inline-flex items-center justify-center px-8 py-4 rounded-xl bg-gradient-to-r from-violet-600 to-cyan-600 text-white font-semibold text-lg shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 transition-shadow"
            >
              Kostenlos starten
            </motion.a>
            <motion.a
              href="https://github.com/open-hallucination-index"
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="inline-flex items-center justify-center px-8 py-4 rounded-xl border border-white/20 bg-white/5 text-neutral-100 font-semibold text-lg hover:bg-white/10 transition-colors"
            >
              GitHub ansehen
            </motion.a>
          </div>
        </div>
      </AnimatedSection>

      {/* Footer spacing */}
      <div className="h-10" />
    </main>
  );
}
