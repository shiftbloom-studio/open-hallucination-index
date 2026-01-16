"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { 
  ChevronDown, 
  ChevronUp, 
  ExternalLink, 
  CheckCircle2, 
  XCircle, 
  Database,
  Globe,
  BookOpen,
  Newspaper,
  FlaskConical,
  Shield,
  Activity
} from "lucide-react";
import { CitationTrace, Evidence, EvidenceSource } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge } from "../ui/badge";

interface CitationTraceViewerProps {
  trace: CitationTrace;
  claimText: string;
}

function getSourceIcon(source: EvidenceSource) {
  const sourceStr = source.toLowerCase();
  
  if (sourceStr.includes("wikipedia") || sourceStr.includes("wikidata") || sourceStr.includes("mediawiki")) {
    return <Globe className="h-3.5 w-3.5" />;
  }
  if (sourceStr.includes("graph") || sourceStr === "graph_exact" || sourceStr === "graph_inferred") {
    return <Database className="h-3.5 w-3.5" />;
  }
  if (sourceStr.includes("pubmed") || sourceStr.includes("ncbi") || sourceStr.includes("clinical") || sourceStr.includes("academic")) {
    return <FlaskConical className="h-3.5 w-3.5" />;
  }
  if (sourceStr.includes("news") || sourceStr === "gdelt") {
    return <Newspaper className="h-3.5 w-3.5" />;
  }
  if (sourceStr.includes("context7") || sourceStr.includes("openalex") || sourceStr.includes("crossref")) {
    return <BookOpen className="h-3.5 w-3.5" />;
  }
  if (sourceStr.includes("osv") || sourceStr.includes("security")) {
    return <Shield className="h-3.5 w-3.5" />;
  }
  return <Activity className="h-3.5 w-3.5" />;
}

function formatSource(source: EvidenceSource): string {
  return source
    .replace(/_/g, " ")
    .split(" ")
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function EvidenceCard({ evidence, type }: { evidence: Evidence; type: "supporting" | "refuting" }) {
  const [expanded, setExpanded] = useState(false);
  
  return (
    <div className={cn(
      "p-3 rounded-lg border transition-colors",
      type === "supporting" 
        ? "bg-green-500/5 border-green-500/20 hover:border-green-500/40" 
        : "bg-red-500/5 border-red-500/20 hover:border-red-500/40"
    )}>
      <div className="flex items-start gap-2 mb-2">
        <div className={cn(
          "p-1.5 rounded",
          type === "supporting" ? "bg-green-500/10" : "bg-red-500/10"
        )}>
          {type === "supporting" ? (
            <CheckCircle2 className="h-4 w-4 text-green-500" />
          ) : (
            <XCircle className="h-4 w-4 text-red-500" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <Badge variant="outline" className="text-xs">
              <span className="mr-1">{getSourceIcon(evidence.source)}</span>
              {formatSource(evidence.source)}
            </Badge>
            {evidence.similarity_score !== null && evidence.similarity_score !== undefined && (
              <span className="text-xs text-muted-foreground">
                Match: {(evidence.similarity_score * 100).toFixed(0)}%
              </span>
            )}
            {evidence.classification_confidence !== null && evidence.classification_confidence !== undefined && (
              <span className="text-xs text-muted-foreground">
                Confidence: {(evidence.classification_confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
          <p className={cn(
            "text-sm leading-relaxed",
            !expanded && "line-clamp-2"
          )}>
            {evidence.content}
          </p>
          {evidence.content.length > 150 && (
            <Button
              variant="ghost"
              size="sm"
              className="h-auto p-0 mt-1 text-xs"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? (
                <>
                  <ChevronUp className="h-3 w-3 mr-1" />
                  Show less
                </>
              ) : (
                <>
                  <ChevronDown className="h-3 w-3 mr-1" />
                  Show more
                </>
              )}
            </Button>
          )}
          {evidence.source_uri && (
            <a
              href={evidence.source_uri}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-xs text-primary hover:underline mt-2"
            >
              <ExternalLink className="h-3 w-3" />
              View source
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

export default function CitationTraceViewer({ trace, claimText }: CitationTraceViewerProps) {
  const [showEvidence, setShowEvidence] = useState(true);
  
  const totalEvidence = trace.supporting_evidence.length + trace.refuting_evidence.length;
  
  return (
    <Card className="border-slate-700/50 bg-slate-800/30">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <CardTitle className="text-base font-medium mb-1">
              Evidence Trail
            </CardTitle>
            <CardDescription className="text-xs">
              {totalEvidence} source{totalEvidence !== 1 ? "s" : ""} analyzed • 
              Strategy: {trace.verification_strategy}
            </CardDescription>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowEvidence(!showEvidence)}
            className="h-8"
          >
            {showEvidence ? (
              <>
                <ChevronUp className="h-4 w-4 mr-1" />
                Hide
              </>
            ) : (
              <>
                <ChevronDown className="h-4 w-4 mr-1" />
                Show
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      
      {showEvidence && (
        <CardContent className="space-y-4">
          {/* Reasoning */}
          <div className="p-3 rounded-lg bg-slate-700/30 border border-slate-600/30">
            <p className="text-xs font-medium text-muted-foreground mb-1">Analysis</p>
            <p className="text-sm leading-relaxed">{trace.reasoning}</p>
          </div>

          {/* Supporting Evidence */}
          {trace.supporting_evidence.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                <h4 className="text-sm font-medium">
                  Supporting Evidence ({trace.supporting_evidence.length})
                </h4>
              </div>
              <div className="space-y-2">
                {trace.supporting_evidence.map((evidence) => (
                  <EvidenceCard 
                    key={evidence.id} 
                    evidence={evidence} 
                    type="supporting" 
                  />
                ))}
              </div>
            </div>
          )}

          {/* Refuting Evidence */}
          {trace.refuting_evidence.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <XCircle className="h-4 w-4 text-red-500" />
                <h4 className="text-sm font-medium">
                  Refuting Evidence ({trace.refuting_evidence.length})
                </h4>
              </div>
              <div className="space-y-2">
                {trace.refuting_evidence.map((evidence) => (
                  <EvidenceCard 
                    key={evidence.id} 
                    evidence={evidence} 
                    type="refuting" 
                  />
                ))}
              </div>
            </div>
          )}

          {/* No Evidence Found */}
          {totalEvidence === 0 && (
            <div className="p-4 rounded-lg bg-yellow-500/5 border border-yellow-500/20 text-center">
              <p className="text-sm text-yellow-500/80">
                No evidence found for this claim
              </p>
            </div>
          )}
        </CardContent>
      )}
    </Card>
  );
}
