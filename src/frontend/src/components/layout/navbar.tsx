"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { StatusDot } from "@/components/common/StatusDot";

const NAV_LINKS = [
  { href: "/verify", label: "Verify" },
  { href: "/calibration", label: "Calibration" },
  { href: "/status", label: "Status" },
  { href: "/about", label: "About" },
] as const;

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b border-[color:var(--border-subtle)] bg-surface-base/85 backdrop-blur-md">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3 group">
          <Image
            src="/logo_black.svg"
            alt="Open Hallucination Index Logo"
            width={36}
            height={34}
            className="transition-transform duration-300 group-hover:scale-105"
            priority
          />
          <span className="font-heading text-base font-semibold tracking-tight text-brand-ink">
            Open Hallucination <span className="text-brand-muted font-normal">Index</span>
          </span>
        </Link>
        <nav className="flex items-center gap-1">
          {NAV_LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "relative rounded-full px-3 py-1.5 text-sm font-medium transition-colors",
                  active
                    ? "text-brand-ink bg-[color:var(--surface-soft)]"
                    : "text-brand-muted hover:text-brand-ink hover:bg-[color:var(--surface-soft)]/70",
                )}
              >
                {link.label}
              </Link>
            );
          })}
          <div className="pl-2">
            <StatusDot />
          </div>
        </nav>
      </div>
    </header>
  );
}
