"use client";

import Link from "next/link";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

export function Navbar() {
  const pathname = usePathname();
  
  const isAuthPage = pathname?.startsWith("/auth");
  const isDashboard = pathname?.startsWith("/dashboard");

  return (
    <header className="border-b border-white/10 relative z-50 backdrop-blur-sm bg-black/40">
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
        <nav className="flex items-center gap-4">
          <Link 
            href="/about" 
            className={cn(
              "text-neutral-300 hover:text-white font-medium transition-colors px-3 py-2",
              pathname === "/about" && "text-white"
            )}
          >
            Über uns
          </Link>
          <Link 
            href="/pricing" 
            className={cn(
              "text-neutral-300 hover:text-white font-medium transition-colors px-3 py-2",
              pathname === "/pricing" && "text-white"
            )}
          >
            Pricing
          </Link>
          {!isDashboard && (
            <>
              <Link href="/auth/login">
                <Button variant="ghost" className="text-neutral-300 hover:text-white font-medium">Login</Button>
              </Link>
              <Link href="/auth/signup">
                <Button className="bg-slate-800 text-white border-slate-700 font-medium hover:bg-slate-700 transition-colors">Sign Up</Button>
              </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
