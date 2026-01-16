"use client";

import { useMemo, useRef, useState, useEffect } from "react";
import { CitationTrace } from "@/lib/api";
import { useTheme } from "next-themes";
import dynamic from "next/dynamic";
import { Card } from "@/components/ui/card";
import { Loader2, Zap } from "lucide-react";
import * as THREE from "three";

// Dynamically import ForceGraph3D to avoid SSR issues
const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[400px] w-full items-center justify-center bg-slate-900/50">
      <div className="flex flex-col items-center gap-2">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="text-sm text-muted-foreground">Initializing 3D Environment...</span>
      </div>
    </div>
  ),
});

interface CitationTraceGraphProps {
  trace: CitationTrace;
  claimText: string;
}

interface GraphNode {
  id: string;
  label: string;
  type: string;
  group: number;
  val: number;
  color: string;
  desc: string;
  status?: string;
  similarity?: number | null;
}

interface GraphLink {
  source: string;
  target: string;
  value: number;
  particleColor: string;
}

export default function CitationTraceGraph({ trace, claimText }: CitationTraceGraphProps) {
  const { theme } = useTheme();
  const graphRef = useRef<any>(null);

  // Transform trace data into graph data
  const graphData = useMemo(() => {
    const nodes: GraphNode[] = [];
    const links: GraphLink[] = [];

    // Central Claim Node
    nodes.push({
      id: "claim",
      label: "Claim",
      type: "claim",
      group: 0,
      val: 20, // Size
      color: "#3b82f6", // Blue
      desc: claimText,
    });

    // Supporting Evidence
    trace.supporting_evidence.forEach((ev, idx) => {
      nodes.push({
        id: ev.id,
        label: ev.source,
        type: "evidence",
        status: "supporting",
        group: 1,
        val: 10,
        color: "#22c55e", // Green
        desc: ev.content,
        similarity: ev.similarity_score,
      });
      links.push({
        source: ev.id,
        target: "claim",
        value: ev.similarity_score ? ev.similarity_score * 5 : 1,
        particleColor: "#22c55e",
      });
    });

    // Refuting Evidence
    trace.refuting_evidence.forEach((ev, idx) => {
      nodes.push({
        id: ev.id,
        label: ev.source,
        type: "evidence",
        status: "refuting",
        group: 2,
        val: 10,
        color: "#ef4444", // Red
        desc: ev.content,
        similarity: ev.similarity_score,
      });
      links.push({
        source: ev.id,
        target: "claim",
        value: ev.similarity_score ? ev.similarity_score * 5 : 1,
        particleColor: "#ef4444",
      });
    });

    return { nodes, links };
  }, [trace, claimText]);

  return (
    <div className="relative h-[400px] w-full overflow-hidden rounded-lg border border-slate-700/50 bg-slate-950 shadow-inner">
        <div className="absolute top-3 right-3 z-10 flex items-center gap-2 rounded-full bg-slate-900/80 px-3 py-1 text-xs backdrop-blur border border-slate-700">
            <Zap className="h-3 w-3 text-yellow-400 fill-yellow-400" />
            <span className="font-medium text-slate-300">Live Neural Net</span>
        </div>
      <ForceGraph3D
        ref={graphRef}
        graphData={graphData}
        backgroundColor="#020617" // slate-950
        nodeColor="color"
        nodeVal="val"
        nodeResolution={16}
        showNavInfo={false}
        
        // Link Styling
        linkWidth={1}
        linkColor={() => "#334155"} // slate-700
        linkDirectionalParticles={4}
        linkDirectionalParticleSpeed={0.005}
        linkDirectionalParticleWidth={2}
        linkDirectionalParticleColor="particleColor"
        
        // Node Object Customization (Glowing Spheres)
        nodeThreeObject={(node: any) => {
            const group = new THREE.Group();
            
            // Core Sphere
            const geometry = new THREE.SphereGeometry(node.val / 2);
            const material = new THREE.MeshPhongMaterial({ 
                color: node.color,
                emissive: node.color,
                emissiveIntensity: 0.6,
                shininess: 100
            });
            const mesh = new THREE.Mesh(geometry, material);
            group.add(mesh);

            // Glow Halo (Transparent Outer Sphere)
            const glowGeometry = new THREE.SphereGeometry((node.val / 2) * 1.4);
            const glowMaterial = new THREE.MeshBasicMaterial({
                color: node.color,
                transparent: true,
                opacity: 0.15,
                side: THREE.BackSide
            });
            const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
            group.add(glowMesh);

            return group;
        }}

        // Tooltip Customization
        nodeLabel={(node: any) => {
            return `
                <div style="
                    background: rgba(15, 23, 42, 0.9); 
                    color: white; 
                    padding: 8px 12px; 
                    border-radius: 6px; 
                    border: 1px solid rgba(51, 65, 85, 0.5);
                    font-family: sans-serif;
                    max-width: 300px;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                ">
                    <div style="font-weight: bold; margin-bottom: 4px; color: ${node.color}">${node.label}</div>
                    <div style="font-size: 0.8em; line-height: 1.4; opacity: 0.9;">${node.desc.substring(0, 100)}${node.desc.length > 100 ? '...' : ''}</div>
                </div>
            `;
        }}
        
        // Initial Camera Position
        onEngineStop={() => graphRef.current.zoomToFit(400)}
        cooldownTicks={100}
      />
    </div>
  );
}
