"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";

export default function SignupPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const router = useRouter();
  const supabase = createClient();

  const getAuthErrorMessage = (message?: string, status?: number) => {
    if (status === 429) {
      return "Too many attempts. Please wait a moment and try again.";
    }
    if (message?.toLowerCase().includes("already registered")) {
      return "An account with this email already exists.";
    }
    return message ?? "Something went wrong. Please try again.";
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;
    setFormError(null);

    if (password !== confirmPassword) {
      setFormError("Passwords do not match.");
      return;
    }

    if (password.length < 6) {
      setFormError("Password must be at least 6 characters.");
      return;
    }

    setLoading(true);

    try {
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          emailRedirectTo: `${window.location.origin}/api/auth/callback`,
        },
      });

      if (error) {
        setFormError(getAuthErrorMessage(error.message, error.status));
        setLoading(false);
      } else {
        toast.success("Check your email to confirm your account!");
        router.push("/auth/login");
      }
    } catch {
      setFormError("An unexpected error occurred. Please try again.");
      setLoading(false);
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-gradient-to-b from-slate-950 via-slate-900 to-background p-4 text-foreground">
      <div className="pointer-events-none absolute inset-0 opacity-60">
        <div className="absolute -left-24 top-12 h-72 w-72 rounded-full bg-primary/20 blur-3xl" />
        <div className="absolute -right-24 bottom-12 h-72 w-72 rounded-full bg-primary/10 blur-3xl" />
      </div>
      <div className="relative mx-auto flex min-h-screen w-full max-w-5xl flex-col items-center justify-center gap-8 md:flex-row">
        <div className="w-full max-w-md space-y-4 text-center md:text-left">
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-primary/80">
            Get started
          </p>
          <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            Create your account in minutes
          </h1>
          <p className="text-sm text-slate-300">
            Join the platform to unlock your personalized dashboard and start tracking insights right away.
          </p>
        </div>
        <Card className="w-full max-w-md border-white/10 bg-white/5 shadow-2xl backdrop-blur">
          <CardHeader>
            <CardTitle className="font-heading tracking-tight text-white">Sign Up</CardTitle>
            <CardDescription className="text-slate-300">
              Create an account to get started
            </CardDescription>
          </CardHeader>
          <form onSubmit={handleSignup} aria-busy={loading} className="space-y-4">
            <CardContent className="space-y-4">
              {formError ? (
                <div
                  role="alert"
                  className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-100"
                >
                  {formError}
                </div>
              ) : null}
              <div className="space-y-2">
                <Label htmlFor="email" className="text-slate-200">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  disabled={loading}
                  className="border-white/10 bg-white/10 text-white placeholder:text-slate-400 focus-visible:ring-primary/60"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password" className="text-slate-200">Password</Label>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  disabled={loading}
                  className="border-white/10 bg-white/10 text-white placeholder:text-slate-400 focus-visible:ring-primary/60"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="confirmPassword" className="text-slate-200">Confirm Password</Label>
                <Input
                  id="confirmPassword"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  disabled={loading}
                  className="border-white/10 bg-white/10 text-white placeholder:text-slate-400 focus-visible:ring-primary/60"
                />
              </div>
            </CardContent>
            <CardFooter className="flex flex-col space-y-4">
              <Button type="submit" className="w-full gap-2" disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Creating account...
                  </>
                ) : (
                  "Sign Up"
                )}
              </Button>
              <p className="text-xs text-slate-300 text-center" aria-live="polite">
                {loading ? "We are creating your account. Please wait..." : "Enter your details to continue."}
              </p>
              <p className="text-sm text-slate-300 text-center">
                Already have an account?{" "}
                <Link href="/auth/login" className="text-primary hover:underline">
                  Login
                </Link>
              </p>
            </CardFooter>
          </form>
        </Card>
      </div>
    </main>
  );
}
