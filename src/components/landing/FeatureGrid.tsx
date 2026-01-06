"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function FeatureGrid() {
  return (
    <section className="relative w-full">
      <div className="mx-auto max-w-7xl px-4 py-20 md:py-28">
        <div className="mx-auto max-w-3xl text-center">
          <p className="text-sm font-medium tracking-wide text-neutral-300">Features</p>
          <h2 className="mt-4 text-3xl font-bold tracking-tight text-neutral-50 md:text-5xl">
            Built for grounded systems.
          </h2>
          <p className="mt-6 text-base leading-relaxed text-neutral-300 md:text-lg">
            A verification-first stack for teams that need reliable outputs, not just fluent ones.
          </p>
        </div>

        <div className="mt-12 grid gap-6 lg:grid-cols-6">
          <Card className="lg:col-span-3 border-white/10 bg-white/5 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-neutral-50">GraphRAG</CardTitle>
            </CardHeader>
            <CardContent className="text-neutral-300">
              Beyond simple vector search—model claims against a structured graph of entities, relations, and citations.
            </CardContent>
          </Card>

          <Card className="lg:col-span-3 border-white/10 bg-white/5 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-neutral-50">Atomic Verification</CardTitle>
            </CardHeader>
            <CardContent className="text-neutral-300">
              Checking facts, not just tokens. Each statement is scored independently so failures are visible and actionable.
            </CardContent>
          </Card>

          <Card className="lg:col-span-6 border-white/10 bg-white/5 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-neutral-50">Open Source</CardTitle>
            </CardHeader>
            <CardContent className="text-neutral-300">
              Transparent and community-driven—inspect the methodology, reproduce results, and contribute improvements.
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}
