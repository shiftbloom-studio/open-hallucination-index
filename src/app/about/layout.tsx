import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Über uns – Open Hallucination Index",
  description: "Erfahren Sie mehr über den Open Hallucination Index: Die erste unabhängige, quelloffene Plattform zur Erkennung und Verifikation von KI-Halluzinationen.",
  openGraph: {
    title: "Über uns – Open Hallucination Index",
    description: "Die Zukunft der KI-Verifikation. Vertrauen durch Transparenz.",
  },
};

export default function AboutLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
