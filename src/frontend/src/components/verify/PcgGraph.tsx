"use client";

import dynamic from "next/dynamic";
import {
  forwardRef,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ClaimVerdict, EdgeType } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";

// react-force-graph-3d uses three.js; dynamic-import to avoid SSR.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ForceGraph3D = dynamic<any>(
  () => import("react-force-graph-3d").then((m) => m.default),
  { ssr: false, loading: () => <PcgGraphSkeleton /> },
);

interface GraphNode {
  id: string;
  label: string;
  pTrue: number;
  informationGain: number;
}

interface GraphLink {
  source: string;
  target: string;
  edgeType: EdgeType;
  strength: number;
}

function bandColor(pTrue: number): string {
  if (pTrue >= 0.8) return "#34d399"; // emerald-400
  if (pTrue >= 0.5) return "#fbbf24"; // amber-400
  return "#f87171"; // rose-400
}

function edgeColor(type: EdgeType): string {
  if (type === "entail") return "rgba(52,211,153,0.55)";
  if (type === "contradict") return "rgba(244,63,94,0.7)";
  return "rgba(148,163,184,0.35)";
}

export function buildGraphData(claims: ClaimVerdict[], hideNeutral: boolean) {
  const nodes: GraphNode[] = claims.map((c) => ({
    id: c.claim.id,
    label: c.claim.text,
    pTrue: c.p_true,
    informationGain: c.information_gain,
  }));

  const linkSet = new Set<string>();
  const links: GraphLink[] = [];
  for (const c of claims) {
    for (const n of c.pcg_neighbors) {
      if (hideNeutral && n.edge_type === "neutral") continue;
      // undirected dedupe
      const [a, b] = [c.claim.id, n.neighbor_claim_id].sort();
      const k = `${a}|${b}|${n.edge_type}`;
      if (linkSet.has(k)) continue;
      linkSet.add(k);
      links.push({
        source: c.claim.id,
        target: n.neighbor_claim_id,
        edgeType: n.edge_type,
        strength: n.edge_strength,
      });
    }
  }
  return { nodes, links };
}

export interface PcgGraphHandle {
  focusNode: (claimId: string) => void;
}

export interface PcgGraphProps {
  claims: ClaimVerdict[];
  height?: number;
  className?: string;
}

function PcgGraphSkeleton() {
  return (
    <div className="flex h-full min-h-[240px] w-full items-center justify-center rounded-lg border border-white/10 bg-slate-950/40 text-xs text-slate-500">
      Loading 3D graph…
    </div>
  );
}

export const PcgGraph = forwardRef<PcgGraphHandle, PcgGraphProps>(function PcgGraph(
  { claims, height = 320, className },
  ref,
) {
  const [hideNeutral, setHideNeutral] = useState(true);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null);

  const data = useMemo(() => buildGraphData(claims, hideNeutral), [claims, hideNeutral]);

  useImperativeHandle(ref, () => ({
    focusNode: (claimId: string) => {
      const node = data.nodes.find((n) => n.id === claimId);
      if (!node) return;
      const g = graphRef.current;
      if (!g) return;
      // cameraPosition(lookAt?, duration) — available on the 3D variant
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (g as any).cameraPosition?.({}, node, 800);
      } catch {
        /* noop */
      }
    },
  }));

  if (claims.length === 0) {
    return (
      <div
        className={cn(
          "flex items-center justify-center rounded-lg border border-dashed border-white/10 text-xs text-slate-500",
          className,
        )}
        style={{ height }}
      >
        Graph appears once claims arrive.
      </div>
    );
  }

  return (
    <div
      className={cn("relative overflow-hidden rounded-lg border border-white/10 bg-slate-950/40", className)}
      style={{ height }}
      data-testid="pcg-graph"
      data-node-count={data.nodes.length}
      data-link-count={data.links.length}
    >
      <div className="absolute left-2 top-2 z-10 flex items-center gap-2 rounded-md bg-black/40 px-2 py-1 text-[10px] text-slate-300 backdrop-blur-sm">
        <label className="flex cursor-pointer items-center gap-1">
          <input
            type="checkbox"
            checked={hideNeutral}
            onChange={(e) => setHideNeutral(e.target.checked)}
          />
          <span>Hide neutral edges</span>
        </label>
        <span className="font-mono text-slate-500">
          {data.nodes.length} nodes · {data.links.length} edges
        </span>
      </div>
      <ForceGraph3D
        ref={graphRef}
        graphData={data}
        nodeLabel={(n: GraphNode) => `${n.label}\n p=${n.pTrue.toFixed(2)}`}
        nodeColor={(n: GraphNode) => bandColor(n.pTrue)}
        nodeVal={(n: GraphNode) => Math.max(1, n.informationGain * 40)}
        linkColor={(l: GraphLink) => edgeColor(l.edgeType)}
        linkWidth={(l: GraphLink) => Math.abs(l.strength) * 2}
        linkOpacity={0.8}
        backgroundColor="rgba(2,6,23,0)"
        showNavInfo={false}
        width={undefined}
        height={height}
      />
    </div>
  );
});
