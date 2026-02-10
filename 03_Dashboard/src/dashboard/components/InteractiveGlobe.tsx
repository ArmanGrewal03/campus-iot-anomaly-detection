import * as React from 'react';
import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import Box from '@mui/material/Box';

import globeLandPoints from '../../data/globeLandPoints.json';

const RADIUS = 1;
const DOT_COLOR = '#8b9dc3'; // light blue-gray – dots = land
const CAMERA_DISTANCE = 2.85;
const AUTO_ROTATE_SPEED = 0.3;

function latLonToXYZ(lat: number, lon: number): [number, number, number] {
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;
  const x = RADIUS * Math.cos(latRad) * Math.cos(lonRad);
  const y = RADIUS * Math.sin(latRad);
  const z = RADIUS * Math.cos(latRad) * Math.sin(lonRad);
  return [x, y, z];
}

/** Convert land points [lat, lon][] from geo data to Float32Array of x,y,z on sphere */
function landPointsToPositions(landPoints: [number, number][]): Float32Array {
  const positions: number[] = [];
  for (const [lat, lon] of landPoints) {
    const [x, y, z] = latLonToXYZ(lat, lon);
    positions.push(x, y, z);
  }
  const out = new Float32Array(positions.length);
  out.set(positions);
  return out;
}

function GlobePoints() {
  const pointsRef = useRef<THREE.Points>(null);
  const positions = useMemo(
    () => landPointsToPositions(globeLandPoints as [number, number][]),
    []
  );
  const pointCount = positions.length / 3;

  useFrame((_state, delta: number) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += AUTO_ROTATE_SPEED * delta;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={pointCount}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.018}
        sizeAttenuation
        color={DOT_COLOR}
        transparent
        opacity={0.92}
      />
    </points>
  );
}

function Scene() {
  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <GlobePoints />
      <OrbitControls
        enableZoom={false}
        enablePan={false}
        minDistance={CAMERA_DISTANCE}
        maxDistance={CAMERA_DISTANCE}
        autoRotate={false}
        rotateSpeed={0.8}
      />
    </>
  );
}

interface InteractiveGlobeProps {
  height?: number;
  seamless?: boolean;
}

export default function InteractiveGlobe({ height = 340, seamless = true }: InteractiveGlobeProps) {
  return (
    <Box
      sx={{
        width: '100%',
        height,
        overflow: 'hidden',
        ...(seamless
          ? { bgcolor: 'transparent', minHeight: height }
          : { borderRadius: 1, bgcolor: 'background.paper', border: '1px solid', borderColor: 'divider' }),
      }}
    >
      <Box sx={{ width: '100%', height }}>
        <Canvas
          gl={{ antialias: true, alpha: true }}
          onCreated={({ gl }) => {
            gl.setClearColor(0x000000, 0);
          }}
          camera={{ position: [0, 0, CAMERA_DISTANCE], fov: 45 }}
          dpr={[1, 2]}
        >
          <Scene />
        </Canvas>
      </Box>
    </Box>
  );
}
