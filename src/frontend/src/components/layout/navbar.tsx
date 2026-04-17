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
    <header className="border-b border-white/10 relative z-50 bg-black/70">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3 group">
          <Image
            src="/logo_white.svg"
            alt="Open Hallucination Index Logo"
            width={40}
            height={38}
            className="transition-transform duration-300 group-hover:scale-105"
          />
          <span className="text-xl font-heading font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-neutral-50 via-neutral-200 to-neutral-400">
            Open Hallucination Index
          </span>
        </Link>
        <nav className="flex items-center gap-2">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "text-sm text-neutral-300 hover:text-white font-medium transition-colors px-3 py-2",
                pathname === link.href && "text-white",
              )}
            >
              {link.label}
            </Link>
          ))}
          <div className="pl-2">
            <StatusDot />
          </div>
        </nav>
      </div>
    </header>
  );
}
