"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { createClient } from "@/lib/supabase/client";
import { createApiClient } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import { useQuery } from "@tanstack/react-query";
import { User } from "@supabase/supabase-js";
import { LogOut, Plus } from "lucide-react";

const AddHallucinationForm = dynamic(() => import("@/components/dashboard/add-hallucination-form"), {
  loading: () => <Card className="mb-8 h-[300px] animate-pulse bg-muted/20" />,
});

interface DashboardClientProps {
  user: User;
}

export default function DashboardClient({ user }: DashboardClientProps) {
  const [isAddingHallucination, setIsAddingHallucination] = useState(false);
  
  // API Settings State
  const [showApiSettings, setShowApiSettings] = useState(false);
  const [apiUrl, setApiUrl] = useState("");
  const [apiStatus, setApiStatus] = useState<"idle" | "checking" | "valid" | "invalid">("idle");
  
  const router = useRouter();
  const supabase = createClient();

  useEffect(() => {
    const savedUrl = localStorage.getItem("open_hallucination_api_url");
    if (savedUrl) {
      setApiUrl(savedUrl);
      validateApi(savedUrl);
    }
  }, []);

  const validateApi = async (url: string) => {
    if (!url) {
      setApiStatus("idle");
      return;
    }
    setApiStatus("checking");
    try {
      // Just a simple check. In a real app we might need a proxy to avoid CORS.
      // We'll perform a fetch and assume if we get a response (even 404) the server is 'reachable'.
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const client = createApiClient(url);
      await client.getHealth();
      
      setApiStatus("valid");
      localStorage.setItem("open_hallucination_api_url", url);
    } catch (error) {
      console.error("API Validation failed:", error);
      setApiStatus("invalid");
    }
  };

  const handleApiUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newUrl = e.target.value;
    setApiUrl(newUrl);
    
    // Debounce validation
    const timeoutId = setTimeout(() => validateApi(newUrl), 800);
    return () => clearTimeout(timeoutId);
  };

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
          <AddHallucinationForm 
            onCancel={() => setIsAddingHallucination(false)}
            onSuccess={() => {
              setIsAddingHallucination(false);
              refetch();
            }}
            showApiSettings={showApiSettings}
            setShowApiSettings={setShowApiSettings}
            apiUrl={apiUrl}
            setApiUrl={setApiUrl}
            apiStatus={apiStatus}
            handleApiUrlChange={handleApiUrlChange}
          />
        )}

        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <Card className="relative overflow-hidden border-none bg-gradient-to-br from-blue-500/10 to-indigo-500/10 dark:from-blue-500/20 dark:to-indigo-500/20 backdrop-blur-md shadow-xl transition-all duration-300 hover:shadow-2xl hover:scale-[1.02] group">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-indigo-600/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <CardHeader className="relative">
              <CardTitle className="bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400 bg-clip-text text-transparent">
                Total Hallucinations
              </CardTitle>
              <CardDescription>All documented cases</CardDescription>
            </CardHeader>
            <CardContent className="relative">
              <p className="text-4xl font-bold bg-gradient-to-r from-blue-700 to-indigo-700 dark:from-blue-300 dark:to-indigo-300 bg-clip-text text-transparent">
                0
              </p>
            </CardContent>
          </Card>

          <Card className="relative overflow-hidden border-none bg-gradient-to-br from-emerald-500/10 to-teal-500/10 dark:from-emerald-500/20 dark:to-teal-500/20 backdrop-blur-md shadow-xl transition-all duration-300 hover:shadow-2xl hover:scale-[1.02] group">
            <div className="absolute inset-0 bg-gradient-to-r from-emerald-600/10 to-teal-600/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <CardHeader className="relative">
              <CardTitle className="bg-gradient-to-r from-emerald-600 to-teal-600 dark:from-emerald-400 dark:to-teal-400 bg-clip-text text-transparent">
                Verified Cases
              </CardTitle>
              <CardDescription>Community verified</CardDescription>
            </CardHeader>
            <CardContent className="relative">
              <p className="text-4xl font-bold bg-gradient-to-r from-emerald-700 to-teal-700 dark:from-emerald-300 dark:to-teal-300 bg-clip-text text-transparent">
                0
              </p>
            </CardContent>
          </Card>

          <Card className="relative overflow-hidden border-none bg-gradient-to-br from-rose-500/10 to-pink-500/10 dark:from-rose-500/20 dark:to-pink-500/20 backdrop-blur-md shadow-xl transition-all duration-300 hover:shadow-2xl hover:scale-[1.02] group">
            <div className="absolute inset-0 bg-gradient-to-r from-rose-600/10 to-pink-600/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <CardHeader className="relative">
              <CardTitle className="bg-gradient-to-r from-rose-600 to-pink-600 dark:from-rose-400 dark:to-pink-400 bg-clip-text text-transparent">
                Critical Severity
              </CardTitle>
              <CardDescription>High priority items</CardDescription>
            </CardHeader>
            <CardContent className="relative">
              <p className="text-4xl font-bold bg-gradient-to-r from-rose-700 to-pink-700 dark:from-rose-300 dark:to-pink-300 bg-clip-text text-transparent">
                0
              </p>
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
