"use client";

import { Shield, Database, TrendingUp } from "lucide-react";

export function LandingFeatures() {
  return (
    <section className="container mx-auto px-4 py-16 relative z-10">
      <div className="grid md:grid-cols-3 gap-8">
        <div className="p-6 rounded-2xl bg-gradient-to-b from-neutral-900 to-black border border-neutral-800 hover:border-neutral-700 transition-colors">
          <Shield className="h-10 w-10 mb-4 text-indigo-500" />
          <h3 className="text-xl font-bold text-neutral-100 mb-2">AI Safety First</h3>
          <p className="text-neutral-400">
            Comprehensive toolkit for identifying and tracking AI hallucinations with precision.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-gradient-to-b from-neutral-900 to-black border border-neutral-800 hover:border-neutral-700 transition-colors">
          <Database className="h-10 w-10 mb-4 text-teal-500" />
          <h3 className="text-xl font-bold text-neutral-100 mb-2">Open Database</h3>
          <p className="text-neutral-400">
            Community-driven repository of verified hallucinations accessiblr via API.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-gradient-to-b from-neutral-900 to-black border border-neutral-800 hover:border-neutral-700 transition-colors">
          <TrendingUp className="h-10 w-10 mb-4 text-rose-500" />
          <h3 className="text-xl font-bold text-neutral-100 mb-2">Analytics & Insights</h3>
          <p className="text-neutral-400">
            Gain valuable insights into hallucination patterns across different models.
          </p>
        </div>
      </div>
    </section>
  );
}
