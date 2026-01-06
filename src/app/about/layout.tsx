import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About Us – Open Hallucination Index",
  description: "Learn more about the Open Hallucination Index: The first independent, open-source platform for detecting and verifying AI hallucinations.",
  openGraph: {
    title: "About Us – Open Hallucination Index",
    description: "The future of AI verification. Trust through transparency.",
  },
};

export default function AboutLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
