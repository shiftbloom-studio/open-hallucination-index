"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Points, PointMaterial } from "@react-three/drei";
import * as THREE from "three";

function KnowledgeGraphPoints() {
  const ref = useRef<THREE.Points>(null);

  const positions = useMemo(() => {
    const count = 900;
    const data = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const theta = 2 * Math.PI * Math.random();
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 1.35 * Math.cbrt(Math.random());

      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta);
      const z = r * Math.cos(phi);

      data[i * 3] = x;
      data[i * 3 + 1] = y;
      data[i * 3 + 2] = z;
    }

    return data;
  }, []);

  useFrame((state, delta) => {
    if (!ref.current) return;

    const targetX = state.mouse.y * 0.25;
    const targetY = state.mouse.x * 0.35;

    ref.current.rotation.x = THREE.MathUtils.lerp(
      ref.current.rotation.x,
      targetX,
      1 - Math.pow(0.001, delta)
    );
    ref.current.rotation.y = THREE.MathUtils.lerp(
      ref.current.rotation.y,
      targetY,
      1 - Math.pow(0.001, delta)
    );

    ref.current.rotation.z += delta * 0.035;
  });

  return (
    <group rotation={[0, 0, Math.PI / 5]}>
      <Points ref={ref} positions={positions} stride={3} frustumCulled={false}>
        <PointMaterial
          transparent
          color="white"
          size={0.018}
          sizeAttenuation
          depthWrite={false}
          opacity={0.55}
        />
      </Points>
    </group>
  );
}

export default function KnowledgeGraphCanvas() {
  return (
    <div className="pointer-events-none absolute inset-0 z-0 opacity-70">
      <Canvas camera={{ position: [0, 0, 1.2] }} dpr={[1, 2]}>
        <ambientLight intensity={0.7} />
        <KnowledgeGraphPoints />
      </Canvas>
    </div>
  );
}
