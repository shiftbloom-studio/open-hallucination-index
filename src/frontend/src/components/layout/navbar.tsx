"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { href: "/verify", label: "Verify" },
  { href: "/calibration", label: "Calibration" },
  { href: "/status", label: "Status" },
  { href: "/about", label: "Studio" },
] as const;

export function Navbar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const updateScrolled = () => setScrolled(window.scrollY > 8);
    updateScrolled();
    window.addEventListener("scroll", updateScrolled, { passive: true });
    return () => window.removeEventListener("scroll", updateScrolled);
  }, []);

  return (
    <header
      className={cn(
        "fixed inset-x-0 top-0 z-50 transition-all duration-500",
        scrolled
          ? "bg-surface-base/90 shadow-[0_4px_24px_rgba(0,0,0,0.05)] backdrop-blur-md"
          : "bg-transparent",
      )}
    >
      <div className="mx-auto flex h-[97px] max-w-7xl items-center justify-between px-6">
        <Link href="/" className="flex items-center space-x-3" aria-label="Open Hallucination Index home">
          <Image
            src="/shiftbloom-logo.png"
            alt="shiftbloom studio Logo"
            width={40}
            height={40}
            className="h-10 w-10 rounded"
            priority
          />
          <span className="text-xl font-semibold tracking-normal text-brand-ink sm:text-2xl">
            shiftbloom <span className="font-light italic text-[color:var(--brand-primary)]">studio.</span>
          </span>
        </Link>

        <nav className="hidden items-center md:flex">
          <div className="flex items-center">
            {NAV_LINKS.map((link) => {
              const active = pathname === link.href || pathname.startsWith(`${link.href}/`);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  aria-current={active ? "page" : undefined}
                  className="nav-link mr-12"
                  data-active={active}
                >
                  {link.label}
                </Link>
              );
            })}
          </div>
          <Link href="/verify" className="tertiary-btn min-h-0 px-6 py-3 text-[15px]">
            Test our work
          </Link>
        </nav>

        <nav className="flex items-center gap-3 md:hidden" aria-label="Mobile navigation">
          <Link href="/verify" className="tertiary-btn min-h-0 px-4 py-2 text-sm">
            Verify
          </Link>
        </nav>
      </div>
    </header>
  );
}
