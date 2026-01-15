"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { 
  ShieldCheck, 
  Loader2, 
  BadgeCheck, 
  AlertTriangle, 
  CheckCircle2, 
  XCircle,
  FileText,
  Coins
} from "lucide-react";
import { toast } from "sonner";
import { ClaimSummary, createApiClient, VerifyTextResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

const CHARACTERS_PER_TOKEN = 1000;
const STANDARD_TARGET_SOURCES = 6;
const EXPERT_TARGET_SOURCES = 18;

interface VerifyAIOutputFormProps {
  userTokens: number;
  onTokensUpdated: (newBalance: number) => void;
}

function extractTrustScore(verificationResult: VerifyTextResponse): number {
  const trustScore =
    verificationResult.trust_score.overall ??
    verificationResult.trust_score.score ??
    0;
  return trustScore;
}

export default function VerifyAIOutputForm({ userTokens, onTokensUpdated }: VerifyAIOutputFormProps) {
  const [text, setText] = useState("");
  const [context, setContext] = useState("");
  const [analysisMode, setAnalysisMode] = useState<"standard" | "expert">("standard");
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<VerifyTextResponse | null>(null);

  const textLength = text.length;
  const tokensNeeded = Math.max(1, Math.ceil(textLength / CHARACTERS_PER_TOKEN));
  const hasEnoughTokens = userTokens >= tokensNeeded;
  const targetSources = analysisMode === "expert" ? EXPERT_TARGET_SOURCES : STANDARD_TARGET_SOURCES;

  const handleVerify = useCallback(async () => {
    if (!text.trim()) {
      toast.error("Please enter text to verify");
      return;
    }

    if (!hasEnoughTokens) {
      toast.error(`Insufficient tokens. You need ${tokensNeeded} tokens but have ${userTokens}.`);
      return;
    }

    setIsVerifying(true);
    setVerificationResult(null);

    try {
      // First, deduct tokens (now secure: server calculates needed amount from actual content)
      const deductResponse = await fetch("/api/tokens", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, context }),
      });

      if (!deductResponse.ok) {
        const error = await deductResponse.json();
        if (deductResponse.status === 402) {
          toast.error(`Insufficient tokens. Need ${error.tokensNeeded}, have ${error.tokensAvailable}.`);
          return;
        }
        throw new Error(error.error || "Failed to deduct tokens");
      }

      const deductResult = await deductResponse.json();
      onTokensUpdated(deductResult.tokensRemaining);

      // Now verify with OHI API
      const client = createApiClient();
      const result = await client.verifyText({
        text,
        context: context || undefined,
        strategy: "adaptive",
        target_sources: targetSources,
      });

      setVerificationResult(result);
      toast.success(`Verification complete! ${deductResult.tokensDeducted} token(s) used.`);
    } catch (error) {
      console.error("Verification failed:", error);
      toast.error("Verification failed. Please check if the OHI API is running.");
    } finally {
      setIsVerifying(false);
    }
  }, [text, context, tokensNeeded, userTokens, hasEnoughTokens, onTokensUpdated, targetSources]);

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case "verified":
      case "supported":
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case "refuted":
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "verified":
      case "supported":
        return "text-green-500 bg-green-500/10 border-green-500/30";
      case "refuted":
        return "text-red-500 bg-red-500/10 border-red-500/30";
      default:
        return "text-yellow-500 bg-yellow-500/10 border-yellow-500/30";
    }
  };

  const getTrustScoreColor = (score: number) => {
    if (score >= 0.8) return "text-green-500";
    if (score >= 0.5) return "text-yellow-500";
    return "text-red-500";
  };

  return (
    <div className="space-y-6">
      <Card className="border-none bg-gradient-to-br from-slate-900/50 to-slate-800/50 backdrop-blur-md shadow-xl">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ShieldCheck className="h-6 w-6 text-primary" />
            Verify AI Output
          </CardTitle>
          <CardDescription>
            Paste your AI-generated text below to verify it for hallucinations and factual accuracy
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Label htmlFor="ai-text">AI Output Text</Label>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <FileText className="h-4 w-4" />
                <span>{textLength.toLocaleString()} characters</span>
                <span className="text-muted-foreground/50">•</span>
                <Coins className="h-4 w-4" />
                <span className={cn(
                  hasEnoughTokens ? "text-green-500" : "text-red-500"
                )}>
                  {tokensNeeded} token{tokensNeeded !== 1 ? "s" : ""} needed
                </span>
              </div>
            </div>
            <textarea
              id="ai-text"
              placeholder="Paste your AI-generated text here for verification..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="flex min-h-[200px] w-full rounded-md border border-input bg-background/50 px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-y"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="context">Context (Optional)</Label>
            <textarea
              id="context"
              placeholder="Add optional context to help with claim disambiguation..."
              value={context}
              onChange={(e) => setContext(e.target.value)}
              className="flex min-h-[80px] w-full rounded-md border border-input bg-background/50 px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-y"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="analysis-mode">Analysis Mode</Label>
            <select
              id="analysis-mode"
              value={analysisMode}
              onChange={(e) => setAnalysisMode(e.target.value as "standard" | "expert")}
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <option value="standard">Standard (target 6 sources)</option>
              <option value="expert">Expert (target 18 sources)</option>
            </select>
            <p className="text-xs text-muted-foreground">
              Sets the preferred number of sources to query for verification.
            </p>
          </div>

          <div className="flex items-center justify-between pt-2">
            <div className="text-sm text-muted-foreground">
              <span className="font-medium">Your balance:</span>{" "}
              <span className={cn(
                "font-bold",
                userTokens > 5 ? "text-green-500" : userTokens > 0 ? "text-yellow-500" : "text-red-500"
              )}>
                {userTokens} token{userTokens !== 1 ? "s" : ""}
              </span>
            </div>
            <Button
              onClick={handleVerify}
              disabled={isVerifying || !text.trim() || !hasEnoughTokens}
              className="min-w-[140px]"
            >
              {isVerifying ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Verifying...
                </>
              ) : (
                <>
                  <ShieldCheck className="h-4 w-4 mr-2" />
                  Verify Text
                </>
              )}
            </Button>
          </div>

          {!hasEnoughTokens && text.length > 0 && (
            <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400">
              <AlertTriangle className="h-4 w-4 inline mr-2" />
              You need {tokensNeeded - userTokens} more token{tokensNeeded - userTokens !== 1 ? "s" : ""} to verify this text.
              <span className="ml-2">
                Come back tomorrow for 5 free tokens!
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {verificationResult && (() => {
        const trustScore = extractTrustScore(verificationResult);
        return (
          <Card className="border-none bg-gradient-to-br from-slate-900/50 to-slate-800/50 backdrop-blur-md shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BadgeCheck className="h-6 w-6 text-primary" />
                Verification Results
              </CardTitle>
              <CardDescription>
                Processed in {verificationResult.processing_time_ms.toFixed(0)}ms
                {verificationResult.cached && " (cached)"}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">{/* Trust Score */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700">
                <p className="text-sm text-muted-foreground mb-1">Trust Score</p>
                <p className={cn(
                  "text-3xl font-bold",
                  getTrustScoreColor(trustScore)
                )}>
                  {(trustScore * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700">
                <p className="text-sm text-muted-foreground mb-1">Claims Analyzed</p>
                <p className="text-3xl font-bold text-primary">
                  {verificationResult.claims.length}
                </p>
              </div>
              <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700">
                <p className="text-sm text-muted-foreground mb-1">Verified</p>
                <p className="text-3xl font-bold text-green-500">
                  {verificationResult.claims.filter((c: ClaimSummary) => 
                    c.status.toLowerCase() === "verified" || c.status.toLowerCase() === "supported"
                  ).length}
                </p>
              </div>
              <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700">
                <p className="text-sm text-muted-foreground mb-1">Issues Found</p>
                <p className="text-3xl font-bold text-red-500">
                  {verificationResult.claims.filter((c: ClaimSummary) => 
                    c.status.toLowerCase() === "refuted"
                  ).length}
                </p>
              </div>
            </div>

            {/* Summary */}
            {verificationResult.summary && (
              <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700">
                <p className="text-sm text-muted-foreground mb-2">Summary</p>
                <p className="text-sm">{verificationResult.summary}</p>
              </div>
            )}

            {/* Claims List */}
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-muted-foreground">Analyzed Claims</h4>
              {verificationResult.claims.map((claim: ClaimSummary) => (
                <div
                  key={claim.id}
                  className={cn(
                    "p-4 rounded-lg border",
                    getStatusColor(claim.status)
                  )}
                >
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5">
                      {getStatusIcon(claim.status)}
                    </div>
                    <div className="flex-1 space-y-2">
                      <p className="text-sm font-medium">{claim.text}</p>
                      <div className="flex items-center gap-4 text-xs text-muted-foreground">
                        <span className="capitalize">{claim.status}</span>
                        <span>Confidence: {(claim.confidence * 100).toFixed(0)}%</span>
                      </div>
                      {claim.reasoning && (
                        <p className="text-xs text-muted-foreground mt-2">
                          {claim.reasoning}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        );
      })()}
    </div>
  );
}
