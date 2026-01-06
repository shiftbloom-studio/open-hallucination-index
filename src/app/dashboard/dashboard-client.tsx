"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { useQuery } from "@tanstack/react-query";
import { User } from "@supabase/supabase-js";
import { LogOut, Plus } from "lucide-react";

interface DashboardClientProps {
  user: User;
}

export default function DashboardClient({ user }: DashboardClientProps) {
  const [isAddingHallucination, setIsAddingHallucination] = useState(false);
  const [content, setContent] = useState("");
  const [source, setSource] = useState("");
  const [severity, setSeverity] = useState("medium");
  const router = useRouter();
  const supabase = createClient();

  const { data: hallucinations, refetch } = useQuery({
    queryKey: ["hallucinations"],
    queryFn: async () => {
      // Placeholder - would fetch from API
      return [];
    },
  });

  const handleLogout = async () => {
    await supabase.auth.signOut();
    toast.success("Logged out successfully");
    router.push("/");
    router.refresh();
  };

  const handleAddHallucination = async (e: React.FormEvent) => {
    e.preventDefault();
    // Placeholder for adding hallucination
    toast.success("Hallucination added successfully!");
    setContent("");
    setSource("");
    setSeverity("medium");
    setIsAddingHallucination(false);
    refetch();
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">{user.email}</span>
            <Button variant="outline" size="sm" onClick={handleLogout}>
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold mb-2">Welcome Back</h2>
            <p className="text-muted-foreground">
              Track and manage AI hallucinations
            </p>
          </div>
          <Button onClick={() => setIsAddingHallucination(!isAddingHallucination)}>
            <Plus className="h-4 w-4 mr-2" />
            Add Hallucination
          </Button>
        </div>

        {isAddingHallucination && (
          <Card className="mb-8">
            <CardHeader>
              <CardTitle>Add New Hallucination</CardTitle>
              <CardDescription>
                Document a new AI hallucination for the index
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleAddHallucination} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="content">Content</Label>
                  <Input
                    id="content"
                    placeholder="Describe the hallucination..."
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="source">Source</Label>
                  <Input
                    id="source"
                    placeholder="AI model or system"
                    value={source}
                    onChange={(e) => setSource(e.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="severity">Severity</Label>
                  <select
                    id="severity"
                    value={severity}
                    onChange={(e) => setSeverity(e.target.value)}
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>
                <div className="flex gap-2">
                  <Button type="submit">Submit</Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setIsAddingHallucination(false)}
                  >
                    Cancel
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        )}

        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <Card>
            <CardHeader>
              <CardTitle>Total Hallucinations</CardTitle>
              <CardDescription>All documented cases</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-4xl font-bold">0</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Verified Cases</CardTitle>
              <CardDescription>Community verified</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-4xl font-bold">0</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Critical Severity</CardTitle>
              <CardDescription>High priority items</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-4xl font-bold">0</p>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Recent Hallucinations</CardTitle>
            <CardDescription>
              Latest documented AI generation errors
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground text-center py-8">
              No hallucinations documented yet. Click &quot;Add Hallucination&quot; to get started.
            </p>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
