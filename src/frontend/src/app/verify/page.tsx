import type { Metadata } from "next";
import { VerifyPage } from "@/components/verify/VerifyPage";

export const metadata: Metadata = {
  title: "Verify — Open Hallucination Index",
  description:
    "Decompose AI-generated text into atomic claims, get calibrated probabilities with uncertainty intervals, and inspect the supporting evidence graph.",
  alternates: { canonical: "/verify" },
};

export default function Page() {
  return <VerifyPage />;
}
