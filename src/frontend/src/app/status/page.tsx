import type { Metadata } from "next";
import { StatusClient } from "./StatusClient";

export const metadata: Metadata = {
  title: "Status — Open Hallucination Index",
  description: "Live per-layer health, latency, and calibration freshness for OHI.",
  alternates: { canonical: "/status" },
};

export default function Page() {
  return <StatusClient />;
}
