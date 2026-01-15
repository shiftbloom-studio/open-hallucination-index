"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Points, PointMaterial } from "@react-three/drei";
import type * as THREE from "three";

function NetworkPoints(props: Record<string, unknown>) {
  const ref = useRef<THREE.Points>(null);
  
  // Generate random points on a sphere
  const sphere = useMemo(() => {
    const points = new Float32Array(500 * 3);
    for (let i = 0; i < 500; i++) {
        const theta = 2 * Math.PI * Math.random();
        const phi = Math.acos(2 * Math.random() - 1);
        const r = 1.2 * Math.cbrt(Math.random()); // Radius ~1.2
        
        const x = r * Math.sin(phi) * Math.cos(theta);
        const y = r * Math.sin(phi) * Math.sin(theta);
        const z = r * Math.cos(phi);
        
        points[i * 3] = x;
        points[i * 3 + 1] = y;
        points[i * 3 + 2] = z;
    }
    return points;
  }, []);

  useFrame((state, delta) => {
    if (ref.current) {
      ref.current.rotation.x -= delta / 10;
      ref.current.rotation.y -= delta / 15;
    }
  });

  return (
    <group rotation={[0, 0, Math.PI / 4]}>
      <Points ref={ref} positions={sphere} stride={3} frustumCulled={false} {...props}>
        <PointMaterial
          transparent
          color="#8b5cf6" // Violet-500
          size={0.02}
          sizeAttenuation={true}
          depthWrite={false}
        />
      </Points>
    </group>
  );
}

export default function NeuralNetworkViz() {
  return (
    <div className="w-full h-full absolute inset-0 z-0 opacity-50">
      <Canvas camera={{ position: [0, 0, 1] }}>
        <NetworkPoints />
      </Canvas>
    </div>
  );
}
