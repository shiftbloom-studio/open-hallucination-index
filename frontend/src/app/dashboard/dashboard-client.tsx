"use client";

import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import dynamic from "next/dynamic";
import { createClient } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import { useQuery } from "@tanstack/react-query";
import { User } from "@supabase/supabase-js";
import { LogOut, Coins, ShieldCheck, TrendingUp, Sparkles } from "lucide-react";
import Link from "next/link";

const VerifyAIOutputForm = dynamic(() => import("@/components/dashboard/verify-ai-output-form"), {
  loading: () => <Card className="mb-8 h-[400px] animate-pulse bg-muted/20" />,
});

interface DashboardClientProps {
  user: User;
}

export default function DashboardClient({ user }: DashboardClientProps) {
  const [userTokens, setUserTokens] = useState<number>(0);
  const router = useRouter();
  const searchParams = useSearchParams();
  const supabase = createClient();

  // Fetch user tokens
  const { data: tokenData, isLoading: tokensLoading, refetch: refetchTokens } = useQuery({
    queryKey: ["user-tokens"],
    queryFn: async () => {
      const res = await fetch("/api/tokens");
      if (!res.ok) throw new Error("Failed to fetch tokens");
      return res.json();
    },
  });

  useEffect(() => {
    if (tokenData?.tokens !== undefined) {
      setUserTokens(tokenData.tokens);
    }
  }, [tokenData]);

  useEffect(() => {
    const success = searchParams.get("success");
    const sessionId = searchParams.get("session_id");

    if (!success) {
      return;
    }

    const checkStatus = async () => {
      if (!sessionId) {
        toast.success("Payment successful! Your tokens will appear shortly.");
        refetchTokens();
        router.replace("/dashboard");
        return;
      }

      try {
        const res = await fetch(`/api/checkout/status?session_id=${sessionId}`);
        if (!res.ok) throw new Error("Failed to fetch checkout status");
        const data = await res.json();

        if (data.paymentStatus === "paid") {
          toast.success("Payment successful! Your tokens are now available.");
          refetchTokens();
        } else {
          toast.error("Payment not completed. If you were charged, contact support.");
        }
      } catch (error) {
        console.error("Failed to verify checkout status:", error);
        toast.error("We couldn't confirm your payment yet. Please refresh later.");
      } finally {
        router.replace("/dashboard");
      }
    };

    checkStatus();
  }, [refetchTokens, router, searchParams]);

  const handleLogout = async () => {
    await supabase.auth.signOut();
    toast.success("Logged out successfully");
    router.push("/");
    router.refresh();
  };

  const handleTokensUpdated = (newBalance: number) => {
    setUserTokens(newBalance);
    refetchTokens();
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">
      <header className="border-b border-slate-800/50 bg-slate-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="text-2xl font-heading font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
              Dashboard
            </Link>
          </div>
          <div className="flex items-center gap-4">
            {/* Token Balance */}
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-amber-500/10 to-orange-500/10 border border-amber-500/20">
              <Coins className="h-4 w-4 text-amber-400" />
              <span className="font-medium text-amber-300">
                {tokensLoading ? "..." : userTokens} tokens
              </span>
            </div>
            <span className="text-sm text-muted-foreground">{user.email}</span>
            <Button variant="outline" size="sm" onClick={handleLogout} className="border-slate-700 hover:border-slate-600">
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-heading font-bold tracking-tight mb-2 bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-300">
            AI Output Verification
          </h2>
          <p className="text-muted-foreground">
            Verify your AI-generated content for hallucinations and factual accuracy
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <Card className="relative overflow-hidden border-none bg-gradient-to-br from-amber-500/10 to-orange-500/10 backdrop-blur-md shadow-xl transition-all duration-300 hover:shadow-2xl hover:scale-[1.02] group">
            <div className="absolute inset-0 bg-gradient-to-r from-amber-600/10 to-orange-600/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <CardHeader className="relative">
              <CardTitle className="bg-gradient-to-r from-amber-400 to-orange-400 bg-clip-text text-transparent flex items-center gap-2">
                <Coins className="h-5 w-5 text-amber-400" />
                Available Tokens
              </CardTitle>
              <CardDescription>Your verification credits</CardDescription>
            </CardHeader>
            <CardContent className="relative">
              <p className="text-4xl font-bold bg-gradient-to-r from-amber-300 to-orange-300 bg-clip-text text-transparent">
                {tokensLoading ? "..." : userTokens}
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                1 token = 1,000 characters
              </p>
            </CardContent>
          </Card>

          <Card className="relative overflow-hidden border-none bg-gradient-to-br from-blue-500/10 to-indigo-500/10 backdrop-blur-md shadow-xl transition-all duration-300 hover:shadow-2xl hover:scale-[1.02] group">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-indigo-600/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <CardHeader className="relative">
              <CardTitle className="bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent flex items-center gap-2">
                <ShieldCheck className="h-5 w-5 text-blue-400" />
                How It Works
              </CardTitle>
              <CardDescription>Simple verification process</CardDescription>
            </CardHeader>
            <CardContent className="relative">
              <ol className="text-sm text-muted-foreground space-y-1">
                <li>1. Paste your AI output</li>
                <li>2. Click verify</li>
                <li>3. Get trust score & claims</li>
              </ol>
            </CardContent>
          </Card>

          <Card className="relative overflow-hidden border-none bg-gradient-to-br from-emerald-500/10 to-teal-500/10 backdrop-blur-md shadow-xl transition-all duration-300 hover:shadow-2xl hover:scale-[1.02] group">
            <div className="absolute inset-0 bg-gradient-to-r from-emerald-600/10 to-teal-600/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <CardHeader className="relative">
              <CardTitle className="bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-emerald-400" />
                API Status
              </CardTitle>
              <CardDescription>Verification service</CardDescription>
            </CardHeader>
            <CardContent className="relative">
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-full bg-green-500 animate-pulse" />
                <span className="text-sm text-green-400">Connected to OHI API</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Verification Form */}
        <VerifyAIOutputForm 
          userTokens={userTokens} 
          onTokensUpdated={handleTokensUpdated}
        />

        {/* Daily Tokens Info */}
        <Card className="mt-8 border-slate-800/50 bg-slate-900/30">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-amber-400" />
              Free Daily Tokens
            </CardTitle>
            <CardDescription>
              Every day you can claim 5 free tokens to verify your AI-generated content.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-muted-foreground">
              <p>Each token allows you to verify up to 1,000 characters of AI-generated text.</p>
              <p className="mt-2 text-amber-300">Come back daily to collect your free tokens!</p>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
