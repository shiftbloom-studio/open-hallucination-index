import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Shield, Sparkles, TrendingUp, Zap } from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b bg-background/80 backdrop-blur">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Sparkles className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-sm uppercase tracking-widest text-muted-foreground">SaaS</p>
              <h1 className="text-2xl font-bold">Open Hallucination Index</h1>
            </div>
          </div>
          <nav className="flex items-center gap-2">
            <ThemeToggle />
            <Link href="/auth/signin">
              <Button variant="ghost">Sign In</Button>
            </Link>
            <Link href="/auth/signup">
              <Button>Get Started</Button>
            </Link>
          </nav>
        </div>
      </header>

      <main className="flex-1">
        <section className="container mx-auto px-4 py-16">
          <div className="grid gap-12 lg:grid-cols-[1.1fr_0.9fr] items-center">
            <div>
              <p className="inline-flex items-center gap-2 rounded-full border bg-background px-4 py-1 text-sm text-muted-foreground">
                <Zap className="h-4 w-4 text-primary" />
                Real-time hallucination checks
              </p>
              <h2 className="text-4xl md:text-5xl font-bold mt-6 leading-tight">
                Make AI outputs safer with instant hallucination scoring
              </h2>
              <p className="text-lg text-muted-foreground mt-4 max-w-xl">
                Analyze any response, measure factual consistency, and create a clearer feedback loop for your teams.
                The Open Hallucination Index gives you a single dashboard to test and track AI reliability.
              </p>
              <div className="mt-8 flex flex-wrap gap-4">
                <Link href="/auth/signup">
                  <Button size="lg">Start free</Button>
                </Link>
                <Link href="/auth/signin">
                  <Button size="lg" variant="outline">Sign in</Button>
                </Link>
              </div>
              <div className="mt-6 text-sm text-muted-foreground">
                No credit card required · Built for teams, researchers, and builders
              </div>
            </div>
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle>What you get</CardTitle>
                <CardDescription>Everything you need to monitor AI accuracy</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4">
                <div className="rounded-lg border bg-muted/40 p-4">
                  <p className="font-semibold">Instant scoring</p>
                  <p className="text-sm text-muted-foreground">
                    Paste text and receive a hallucination score in seconds.
                  </p>
                </div>
                <div className="rounded-lg border bg-muted/40 p-4">
                  <p className="font-semibold">Actionable insights</p>
                  <p className="text-sm text-muted-foreground">
                    View structured feedback and build validation workflows.
                  </p>
                </div>
                <div className="rounded-lg border bg-muted/40 p-4">
                  <p className="font-semibold">Secure access</p>
                  <p className="text-sm text-muted-foreground">
                    Supabase authentication keeps teams and data protected.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        <section className="container mx-auto px-4 pb-16">
          <div className="grid md:grid-cols-3 gap-8">
            <Card>
              <CardHeader>
                <Shield className="h-10 w-10 mb-4 text-primary" />
                <CardTitle>AI safety first</CardTitle>
                <CardDescription>
                  Comprehensive toolkit for identifying hallucinations
                </CardDescription>
              </CardHeader>
              <CardContent>
                Monitor and improve factual accuracy with a consistent, repeatable evaluation flow.
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <TrendingUp className="h-10 w-10 mb-4 text-primary" />
                <CardTitle>Operational visibility</CardTitle>
                <CardDescription>
                  Confidence signals your team can trust
                </CardDescription>
              </CardHeader>
              <CardContent>
                Share results and create guardrails across product, research, and policy teams.
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <Sparkles className="h-10 w-10 mb-4 text-primary" />
                <CardTitle>Built for scale</CardTitle>
                <CardDescription>
                  Fast checks, clear output, ready to integrate
                </CardDescription>
              </CardHeader>
              <CardContent>
                Use the Open Hallucination Index API to score content in real time.
              </CardContent>
            </Card>
          </div>
        </section>
      </main>

      <footer className="border-t py-8">
        <div className="container mx-auto px-4 text-center text-muted-foreground">
          <p>&copy; 2024 Open Hallucination Index. Open source and community-driven.</p>
        </div>
      </footer>
    </div>
  );
}
