"use client";

import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import { useMutation } from "@tanstack/react-query";
import { User } from "@supabase/supabase-js";
import { LogOut, Sparkles } from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { ThemeToggle } from "@/components/theme-toggle";

const formSchema = z.object({
  text: z
    .string()
    .min(20, "Please enter at least 20 characters for a reliable check."),
});

type FormValues = z.infer<typeof formSchema>;

type HallucinationResponse = {
  score?: number;
  label?: string;
  details?: string;
  [key: string]: unknown;
};

interface DashboardClientProps {
  user: User;
}

export default function DashboardClient({ user }: DashboardClientProps) {
  const router = useRouter();
  const supabase = createClient();
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      text: "",
    },
  });

  const apiBaseUrl = process.env.NEXT_PUBLIC_OHI_API_URL ?? "http://localhost:8080";

  const mutation = useMutation<HallucinationResponse, Error, FormValues>({
    mutationFn: async (values) => {
      const response = await fetch(`${apiBaseUrl}/hallucinations/check`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: values.text }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Unable to analyze the text.");
      }

      return (await response.json()) as HallucinationResponse;
    },
    onSuccess: () => {
      toast.success("Analysis complete");
    },
    onError: (error) => {
      toast.error(error.message);
    },
  });

  const handleLogout = async () => {
    await supabase.auth.signOut();
    toast.success("Logged out successfully");
    router.push("/");
    router.refresh();
  };

  const handleSubmit = form.handleSubmit((values) => {
    mutation.mutate(values);
  });

  const score =
    typeof mutation.data?.score === "number" ? mutation.data?.score : null;

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b bg-background/80 backdrop-blur">
        <div className="container mx-auto px-4 py-4 flex flex-wrap gap-4 items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Sparkles className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Dashboard</p>
              <h1 className="text-2xl font-bold">Hallucination Check</h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">{user.email}</span>
            <ThemeToggle />
            <Button variant="outline" size="sm" onClick={handleLogout}>
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-8">
        <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <Card className="h-fit">
            <CardHeader>
              <CardTitle>Run a new check</CardTitle>
              <CardDescription>
                Paste the AI-generated text below to analyze factual consistency.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="text">Text to analyze</Label>
                  <Textarea
                    id="text"
                    placeholder="Paste or type the response you want to evaluate..."
                    rows={10}
                    {...form.register("text")}
                  />
                  {form.formState.errors.text && (
                    <p className="text-sm text-destructive">
                      {form.formState.errors.text.message}
                    </p>
                  )}
                </div>
                <div className="flex flex-wrap items-center gap-3">
                  <Button type="submit" disabled={mutation.isPending}>
                    {mutation.isPending ? "Checking..." : "Check"}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => form.reset()}
                    disabled={mutation.isPending}
                  >
                    Reset
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>

          <Card className="h-fit">
            <CardHeader>
              <CardTitle>Latest result</CardTitle>
              <CardDescription>
                Review the score and supporting details returned by the API.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {mutation.isPending && (
                <div className="rounded-lg border bg-muted/40 p-4 text-sm text-muted-foreground">
                  Analyzing your text. This usually takes a few seconds.
                </div>
              )}
              {!mutation.isPending && !mutation.data && (
                <div className="rounded-lg border bg-muted/40 p-4 text-sm text-muted-foreground">
                  Submit a passage to generate a hallucination score.
                </div>
              )}
              {mutation.data && (
                <div className="space-y-4">
                  <div className="rounded-lg border bg-muted/40 p-4">
                    <p className="text-sm text-muted-foreground">Score</p>
                    <p className="text-3xl font-bold">
                      {score !== null ? score.toFixed(2) : "N/A"}
                    </p>
                    {mutation.data.label && (
                      <p className="text-sm text-muted-foreground mt-1">
                        {mutation.data.label}
                      </p>
                    )}
                  </div>
                  <div className="rounded-lg border bg-muted/40 p-4">
                    <p className="text-sm font-semibold mb-2">Raw response</p>
                    <pre className="text-xs whitespace-pre-wrap text-muted-foreground">
                      {JSON.stringify(mutation.data, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
