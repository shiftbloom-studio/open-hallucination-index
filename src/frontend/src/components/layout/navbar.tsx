"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { usePathname, useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import { createClient } from "@/lib/supabase/client";
import { type AuthChangeEvent, type Session, type User } from "@supabase/supabase-js";
import { Coins, LogOut, LayoutDashboard } from "lucide-react";
import { toast } from "sonner";

export function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const supabase = createClient();

  const [user, setUser] = useState<User | null>(null);
  const [tokens, setTokens] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);


  const isDashboard = pathname?.startsWith("/dashboard");

  useEffect(() => {
    const fetchTokens = async () => {
      try {
        const res = await fetch("/api/tokens");
        if (res.ok) {
          const data = await res.json();
          setTokens(data.tokens);
        } else {
          setTokens(null);
        }
      } catch (error) {
        console.error("Failed to fetch tokens:", error);
        setTokens(null);
      }
    };

    const getUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      setUser(user);
      setLoading(false);

      if (user) {
        await fetchTokens();
      }
    };
    getUser();

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event: AuthChangeEvent, session: Session | null) => {
      setUser(session?.user ?? null);
      setLoading(false);
      if (session?.user) {
        fetchTokens();
      } else {
        setTokens(null);
      }
    });

    return () => subscription.unsubscribe();
  }, [supabase.auth]);

  const handleLogout = async () => {
    await supabase.auth.signOut();
    setUser(null);
    setTokens(null);
    toast.success("Logged out successfully");
    router.push("/");
    router.refresh();
  };

  // Don't render navbar on dashboard (it has its own header)
  if (isDashboard) {
    return null;
  }

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
        <nav className="flex items-center gap-4">
          <Link 
            href="/pricing" 
            className={cn(
              "text-neutral-300 hover:text-white font-medium transition-colors px-3 py-2",
              pathname === "/pricing" && "text-white"
            )}
          >
            Pricing
          </Link>
          <Link 
            href="/about" 
            className={cn(
              "text-neutral-300 hover:text-white font-medium transition-colors px-3 py-2",
              pathname === "/about" && "text-white"
            )}
          >
            About
          </Link>
          {user ? (
            <>
              {/* Token Balance */}
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-gradient-to-r from-amber-500/10 to-orange-500/10 border border-amber-500/20">
                <Coins className="h-4 w-4 text-amber-400" />
                <span className="font-medium text-amber-300 text-sm">
                  {tokens !== null ? tokens : "..."} tokens
                </span>
              </div>
              
              {/* Dashboard Link */}
              <Link href="/dashboard">
                <Button variant="ghost" size="sm" className="text-neutral-300 hover:text-white">
                  <LayoutDashboard className="h-4 w-4 mr-2" />
                  Dashboard
                </Button>
              </Link>
              
              {/* User Email & Logout */}
              <span className="text-sm text-muted-foreground hidden md:inline">{user.email}</span>
              <Button variant="outline" size="sm" onClick={handleLogout} className="border-slate-700 hover:border-slate-600">
                <LogOut className="h-4 w-4 mr-2" />
                Logout
              </Button>
            </>
          ) : (
            <>
              <Link 
                href="/auth/login" 
                className={cn(
                  "text-neutral-300 hover:text-white font-medium transition-colors px-3 py-2 cursor-pointer",
                  pathname === "/auth/login" && "text-white"
                )}
              >
                Login
              </Link>
              <Link href="/auth/signup">
                <Button className="bg-slate-800 text-white border border-slate-700 font-medium hover:bg-slate-700 transition-colors cursor-pointer">Sign Up</Button>
              </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
