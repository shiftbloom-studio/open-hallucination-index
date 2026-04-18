"use client";

import Link from "next/link";
import { openCookieSettings } from "./cookie-consent";

const LEGAL_LINKS = [
  { href: "/impressum", label: "Legal Notice" },
  { href: "/agb", label: "Terms of Service" },
  { href: "/datenschutz", label: "Privacy Policy" },
  { href: "/cookies", label: "Cookie Policy" },
  { href: "/eula", label: "EULA" },
  { href: "/disclaimer", label: "Disclaimer" },
  { href: "/accessibility", label: "Accessibility" },
] as const;

export function Footer() {
  return (
    <footer className="border-t border-[color:var(--border-subtle)] py-10 relative z-10 bg-surface-base">
      <div className="container mx-auto px-4 text-center space-y-4">
        <nav className="flex flex-wrap justify-center gap-x-6 gap-y-2 text-sm">
          {LEGAL_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="text-brand-muted hover:text-brand-ink transition-colors"
            >
              {link.label}
            </Link>
          ))}
          <button
            onClick={openCookieSettings}
            className="text-brand-muted hover:text-brand-ink transition-colors"
          >
            Cookie Settings
          </button>
        </nav>
        <p className="text-xs text-brand-subtle">
          &copy; {new Date().getFullYear()} Open Hallucination Index.
        </p>
      </div>
    </footer>
  );
}
