"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Settings, Check, X, Loader2, ShieldCheck, BadgeCheck } from "lucide-react";
import { toast } from "sonner";
import { createApiClient, VerifyTextResponse } from "@/lib/api";

interface AddHallucinationFormProps {
  onCancel: () => void;
  onSuccess: () => void;
  // Props for API Settings state (could also be moved to context/store, but passing down for now)
  showApiSettings: boolean;
  setShowApiSettings: (show: boolean) => void;
  apiUrl: string;
  setApiUrl: (url: string) => void;
  apiStatus: "idle" | "checking" | "valid" | "invalid";
  handleApiUrlChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export default function AddHallucinationForm({
  onCancel,
  onSuccess,
  showApiSettings,
  setShowApiSettings,
  apiUrl,
  handleApiUrlChange,
  apiStatus
}: AddHallucinationFormProps) {
  const [content, setContent] = useState("");
  const [source, setSource] = useState("");
  const [severity, setSeverity] = useState("medium");
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<VerifyTextResponse | null>(null);

  const handleVerify = async () => {
    if (!apiUrl) {
      toast.error("Please configure API URL first");
      setShowApiSettings(true);
      return;
    }
    if (!content) {
      toast.error("Please enter content to verify");
      return;
    }

    setIsVerifying(true);
    try {
      const client = createApiClient(apiUrl);
      const result = await client.verifyText({
        text: content,
        context: source || undefined,
        strategy: "hybrid" // Default strategy
      });
      setVerificationResult(result);
      toast.success("Verification complete");
    } catch (error) {
      console.error(error);
      toast.error("Verification failed");
    } finally {
      setIsVerifying(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // Placeholder for adding hallucination
    toast.success("Hallucination added successfully!");
    setContent("");
    setSource("");
    setSeverity("medium");
    onSuccess();
  };

  return (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle>Add New Hallucination</CardTitle>
        <CardDescription>
          Document a new AI hallucination for the index
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2 relative">
            <div className="flex justify-between items-center">
              <Label htmlFor="content">Content</Label>
              <div className="relative">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6"
                  onClick={() => setShowApiSettings(!showApiSettings)}
                >
                  <Settings className="h-4 w-4" />
                </Button>
                
                {showApiSettings && (
                  <Card className="absolute right-0 top-8 z-50 w-80 shadow-lg border-2">
                    <CardHeader className="py-3 px-4">
                      <CardTitle className="text-sm font-medium">API Configuration</CardTitle>
                    </CardHeader>
                    <CardContent className="py-2 px-4 pb-4 space-y-2">
                        <div className="space-y-1">
                          <Label htmlFor="apiUrl" className="text-xs">Open Hallucination API URL</Label>
                          <div className="relative">
                            <Input
                              id="apiUrl"
                              placeholder="https://api.example.com"
                              value={apiUrl}
                              onChange={handleApiUrlChange}
                              className="pr-8 h-8 text-sm"
                            />
                            <div className="absolute right-2 top-1/2 -translate-y-1/2">
                              {apiStatus === "checking" && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
                              {apiStatus === "valid" && <Check className="h-4 w-4 text-green-500" />}
                              {apiStatus === "invalid" && <X className="h-4 w-4 text-red-500" />}
                            </div>
                          </div>
                          {apiStatus === "invalid" && <p className="text-xs text-red-500">API not reachable</p>}
                          {apiStatus === "valid" && <p className="text-xs text-green-500">API connected</p>}
                        </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
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
          
          {verificationResult && (
            <div className="mt-4 p-4 rounded-lg bg-muted/50 border">
              <div className="flex items-center gap-2 mb-2">
                <BadgeCheck className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">Verification Result</h3>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Trust Score</p>
                  <p className="font-medium text-lg">{(verificationResult.trust_score.score * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Claims Verified</p>
                  <p className="font-medium text-lg">{verificationResult.claims.length}</p>
                </div>
              </div>
              {verificationResult.summary && (
                <div className="mt-3">
                  <p className="text-muted-foreground text-xs uppercase mb-1">Summary</p>
                  <p className="text-sm">{verificationResult.summary}</p>
                </div>
              )}
            </div>
          )}

          <div className="flex gap-2 pt-2">
            <Button 
              type="button" 
              variant="secondary" 
              onClick={handleVerify}
              disabled={isVerifying || !content}
            >
              {isVerifying ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <ShieldCheck className="h-4 w-4 mr-2" />}
              Verify
            </Button>
            <Button type="submit">Submit</Button>
            <Button
              type="button"
              variant="outline"
              onClick={onCancel}
            >
              Cancel
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
