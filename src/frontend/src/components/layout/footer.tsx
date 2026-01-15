"use client";

import Link from "next/link";
import { openCookieSettings } from "./cookie-consent";

export function Footer() {
  return (
    <footer className="border-t border-white/10 py-10 relative z-10 bg-black">
      <div className="container mx-auto px-4 text-center text-neutral-500 space-y-4">
        <nav className="flex flex-wrap justify-center gap-x-6 gap-y-2 text-sm">
          <Link className="hover:text-neutral-200" href="/impressum">Legal Notice</Link>
          <Link className="hover:text-neutral-200" href="/agb">Terms of Service</Link>
          <Link className="hover:text-neutral-200" href="/datenschutz">Privacy Policy</Link>
          <Link className="hover:text-neutral-200" href="/cookies">Cookie Policy</Link>
          <button
            onClick={openCookieSettings}
            className="hover:text-neutral-200 transition-colors"
          >
            Cookie Settings
          </button>
          <Link className="hover:text-neutral-200" href="/eula">EULA</Link>
          <Link className="hover:text-neutral-200" href="/disclaimer">Disclaimer</Link>
          <Link className="hover:text-neutral-200" href="/accessibility">Accessibility</Link>
        </nav>
        <p className="text-xs">&copy; {new Date().getFullYear()} Open Hallucination Index.</p>
      </div>
    </footer>
  );
}
