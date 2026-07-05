"use client";

import Link from "next/link";
import { openCookieSettings } from "./cookie-consent";

const PRIMARY_LINKS = [
  { href: "/verify", label: "Verify" },
  { href: "/calibration", label: "Calibration" },
  { href: "/status", label: "Status" },
  { href: "/about", label: "Studio" },
] as const;

const LEGAL_LINKS = [
  { href: "/impressum", label: "Legal Notice" },
  { href: "/agb", label: "Terms" },
  { href: "/datenschutz", label: "Privacy" },
  { href: "/cookies", label: "Cookies" },
  { href: "/eula", label: "EULA" },
  { href: "/disclaimer", label: "Disclaimer" },
  { href: "/accessibility", label: "Accessibility" },
] as const;

export function Footer() {
  return (
    <footer className="relative z-10 overflow-hidden bg-[color:var(--brand-dark)] py-20 text-white">
      <div className="absolute inset-x-0 top-0 h-px bg-white/10" />
      <div className="absolute inset-y-0 left-[18%] w-px bg-white/[0.04]" />

      <div className="container relative z-10 mx-auto px-6">
        <div className="mx-auto max-w-6xl">
          <div className="mb-16 grid gap-8 border-b border-white/10 pb-14 lg:grid-cols-12 lg:items-end">
            <div className="lg:col-span-7">
              <span className="mb-5 block font-mono text-xs uppercase tracking-[0.24em] text-[color:var(--brand-primary)]">
                Shiftbloom Studio
              </span>
              <h2 className="text-[2.65rem] font-semibold leading-[1.03] tracking-normal text-white md:text-[4rem] lg:text-[4.75rem]">
                Open work, from Hamburg.
              </h2>
            </div>
            <div className="lg:col-span-4 lg:col-start-9">
              <p className="text-[1.05rem] font-light leading-relaxed text-white/48 md:text-[1.15rem]">
                Code projects, factuality tools, and collective digital practice. Quietly open,
                intentionally inspectable.
              </p>
              <a
                href="mailto:hello@shiftbloom.studio"
                className="mt-6 inline-flex rounded-full border border-white/14 px-6 py-3 text-base font-semibold text-white transition-colors hover:border-[color:var(--brand-primary)]/60 hover:text-[color:var(--brand-primary)]"
              >
                hello@shiftbloom.studio
              </a>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-10 md:grid-cols-3">
            <div>
              <h3 className="mb-5 text-sm font-semibold uppercase tracking-[0.18em] text-white/78">Work</h3>
              <nav className="flex flex-col gap-3 text-white/45">
                {PRIMARY_LINKS.map((link) => (
                  <Link key={link.href} href={link.href} className="hover:text-[color:var(--brand-primary)]">
                    {link.label}
                  </Link>
                ))}
                <a
                  href="https://github.com/shiftbloom-studio/open-hallucination-index"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-[color:var(--brand-primary)]"
                >
                  GitHub
                </a>
              </nav>
            </div>

            <div>
              <h3 className="mb-5 text-sm font-semibold uppercase tracking-[0.18em] text-white/78">Contact</h3>
              <ul className="space-y-3 text-white/45">
                <li>
                  <a href="mailto:fabian@shiftbloom.studio" className="hover:text-[color:var(--brand-primary)]">
                    fabian@shiftbloom.studio
                  </a>
                </li>
                <li>Hamburg, Germany</li>
                <li>
                  <a
                    href="https://github.com/shiftbloom-studio"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-[color:var(--brand-primary)]"
                  >
                    GitHub: shiftbloom-studio
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="mb-5 text-sm font-semibold uppercase tracking-[0.18em] text-white/78">Housekeeping</h3>
              <nav className="flex flex-col items-start gap-3 text-white/45">
                {LEGAL_LINKS.map((link) => (
                  <Link key={link.href} href={link.href} className="hover:text-[color:var(--brand-primary)]">
                    {link.label}
                  </Link>
                ))}
                <button
                  type="button"
                  onClick={openCookieSettings}
                  className="text-left hover:text-[color:var(--brand-primary)]"
                >
                  Cookie Settings
                </button>
              </nav>
            </div>
          </div>
          <div className="mt-14 border-t border-white/10 pt-8 text-sm text-white/35">
            &copy; {new Date().getFullYear()} shiftbloom studio. Born in Hamburg.
          </div>
        </div>
      </div>
    </footer>
  );
}
