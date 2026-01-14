"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ShieldCheck, BadgeCheck, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { createApiClient, VerifyTextResponse } from "@/lib/api";

interface AddHallucinationFormProps {
  onCancel: () => void;
  onSuccess: () => void;
}

export default function AddHallucinationForm({
  onCancel,
  onSuccess,
}: AddHallucinationFormProps) {
  const [content, setContent] = useState("");
  const [source, setSource] = useState("");
  const [severity, setSeverity] = useState("medium");
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<VerifyTextResponse | null>(null);

  const handleVerify = async () => {
    if (!content) {
      toast.error("Please enter content to verify");
      return;
    }

    setIsVerifying(true);
    try {
      const client = createApiClient();
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
                  <p className="font-medium text-lg">{((verificationResult.trust_score.overall ?? verificationResult.trust_score.score ?? 0) * 100).toFixed(1)}%</p>
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
