import type { Metadata } from "next";
import { CalibrationClient } from "./CalibrationClient";

export const metadata: Metadata = {
  title: "Calibration — Open Hallucination Index",
  description: "Empirical coverage and interval widths per domain.",
  alternates: { canonical: "/calibration" },
};

export default function Page() {
  return <CalibrationClient />;
}
