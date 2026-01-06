import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Shield, Database, TrendingUp } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold">Open Hallucination Index</h1>
          <nav className="flex gap-4">
            <Link href="/auth/login">
              <Button variant="ghost">Login</Button>
            </Link>
            <Link href="/auth/signup">
              <Button>Sign Up</Button>
            </Link>
          </nav>
        </div>
      </header>

      <main className="flex-1">
        <section className="container mx-auto px-4 py-16 text-center">
          <h2 className="text-5xl font-bold mb-6">
            Enhancing AI Safety Through Transparency
          </h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            An open-source initiative dedicated to measuring factual consistency and mitigating generation errors in modern Generative AI architectures.
          </p>
          <div className="flex gap-4 justify-center">
            <Link href="/auth/signup">
              <Button size="lg">Get Started</Button>
            </Link>
            <Link href="/dashboard">
              <Button size="lg" variant="outline">View Dashboard</Button>
            </Link>
          </div>
        </section>

        <section className="container mx-auto px-4 py-16">
          <div className="grid md:grid-cols-3 gap-8">
            <Card>
              <CardHeader>
                <Shield className="h-10 w-10 mb-4 text-primary" />
                <CardTitle>AI Safety First</CardTitle>
                <CardDescription>
                  Comprehensive toolkit for identifying and tracking AI hallucinations
                </CardDescription>
              </CardHeader>
              <CardContent>
                Monitor and improve the factual accuracy of AI-generated content with our advanced detection system.
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <Database className="h-10 w-10 mb-4 text-primary" />
                <CardTitle>Open Database</CardTitle>
                <CardDescription>
                  Community-driven repository of verified hallucinations
                </CardDescription>
              </CardHeader>
              <CardContent>
                Access and contribute to an ever-growing database of documented AI generation errors.
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <TrendingUp className="h-10 w-10 mb-4 text-primary" />
                <CardTitle>Analytics & Insights</CardTitle>
                <CardDescription>
                  Track trends and patterns in AI model behavior
                </CardDescription>
              </CardHeader>
              <CardContent>
                Gain valuable insights into hallucination patterns across different models and use cases.
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
