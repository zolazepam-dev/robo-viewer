import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Grid } from '@react-three/drei';
import * as THREE from 'three';

interface RobotState {
  id: number;
  position: { x: number; y: number; z: number };
  hp: number;
}

interface Visualizer3DProps {
  robotStates: RobotState[];
}

function Robot({ state }: { state: RobotState }) {
  const meshRef = useRef<THREE.Group>(null);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.position.set(state.position.x, state.position.y, state.position.z);
    }
  });

  return (
    <group ref={meshRef}>
      {/* Core */}
      <mesh castShadow>
        <sphereGeometry args={[0.5, 32, 32]} />
        <meshStandardMaterial color={state.id === 0 ? "cyan" : "red"} />
      </mesh>
      {/* Visual Health Indicator */}
      <mesh position={[0, 1, 0]}>
        <boxGeometry args={[state.hp / 100, 0.1, 0.1]} />
        <meshBasicMaterial color="green" />
      </mesh>
    </group>
  );
}

export function Visualizer3D({ robotStates }: Visualizer3DProps) {
  return (
    <div style={{ width: '100%', height: '500px', background: '#111', borderRadius: '8px', overflow: 'hidden' }}>
      <Canvas shadows camera={{ position: [20, 20, 20], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} castShadow />
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        <Grid infiniteGrid fadeDistance={50} cellColor="#444" sectionColor="#666" />
        
        {robotStates.map((state) => (
          <Robot key={state.id} state={state} />
        ))}

        <OrbitControls />
      </Canvas>
    </div>
  );
}
