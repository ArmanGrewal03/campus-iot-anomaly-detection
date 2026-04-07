import * as React from 'react';
import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';
import * as THREE from 'three';
import Box from '@mui/material/Box';

import globeLandPoints from '../../data/globeLandPoints.json';

const RADIUS = 1;
const DOT_COLOR = '#8b9dc3'; // light blue-gray – dots = land
const ROUTE_COLOR_BLUE = '#58b8ff';
const ROUTE_COLOR_RED = '#ff7391';
const PACKET_COLOR_BLUE = '#86d2ff';
const PACKET_COLOR_RED = '#ff89a6';
const CAMERA_DISTANCE = 2.8;
const AUTO_ROTATE_SPEED = 0.3;
const TRAIL_SEGMENTS = 6;

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
  const positions = useMemo(
    () => landPointsToPositions(globeLandPoints as [number, number][]),
    []
  );
  const pointCount = positions.length / 3;

  return (
    <points>
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

type RouteDef = {
  curve: THREE.QuadraticBezierCurve3;
  speed: number;
  phase: number;
};

function createRoute(start: [number, number], end: [number, number], phase: number): RouteDef {
  const [sx, sy, sz] = latLonToXYZ(start[0], start[1]);
  const [ex, ey, ez] = latLonToXYZ(end[0], end[1]);

  const startVec = new THREE.Vector3(sx, sy, sz).multiplyScalar(1.002);
  const endVec = new THREE.Vector3(ex, ey, ez).multiplyScalar(1.002);
  const mid = startVec.clone().add(endVec).multiplyScalar(0.5).normalize().multiplyScalar(RADIUS + 0.055);

  return {
    curve: new THREE.QuadraticBezierCurve3(startVec, mid, endVec),
    speed: 0.12 + (phase % 3) * 0.015,
    phase,
  };
}

function DataPacketRoutes() {
  const routePairs: Array<[[number, number], [number, number]]> = useMemo(
    () => [
      [[43.65107, -79.347015], [40.7128, -74.006]],
      [[40.7128, -74.006], [51.5074, -0.1278]],
      [[51.5074, -0.1278], [48.8566, 2.3522]],
      [[35.6762, 139.6503], [1.3521, 103.8198]],
      [[1.3521, 103.8198], [22.3193, 114.1694]],
      [[-33.8688, 151.2093], [35.6762, 139.6503]],
      [[52.52, 13.405], [43.65107, -79.347015]],
      [[37.7749, -122.4194], [35.6762, 139.6503]],
      [[-23.5505, -46.6333], [40.7128, -74.006]],
      [[28.6139, 77.209], [51.5074, -0.1278]],
      [[34.0522, -118.2437], [-33.9249, 18.4241]],
      [[55.7558, 37.6173], [35.6895, 139.6917]],
      [[19.4326, -99.1332], [41.0082, 28.9784]],
      [[-34.6037, -58.3816], [52.3676, 4.9041]],
      [[25.2048, 55.2708], [-1.2921, 36.8219]],
      [[59.3293, 18.0686], [60.1699, 24.9384]],
      [[31.2304, 121.4737], [37.5665, 126.978]],
    ],
    []
  );

  const routes = useMemo(
    () => routePairs.map((pair, idx) => createRoute(pair[0], pair[1], idx * 0.11)),
    [routePairs]
  );

  const packetBlueRefs = useRef<Array<THREE.Mesh | null>>([]);
  const packetRedRefs = useRef<Array<THREE.Mesh | null>>([]);
  const blueTrailRefs = useRef<Array<Array<THREE.Mesh | null>>>([]);
  const redTrailRefs = useRef<Array<Array<THREE.Mesh | null>>>([]);

  const wrap01 = (value: number): number => {
    if (value >= 0) return value % 1;
    return (value % 1 + 1) % 1;
  };

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    routes.forEach((route, idx) => {
      const bluePacket = packetBlueRefs.current[idx];
      const redPacket = packetRedRefs.current[idx];

      const blueProgress = (t * route.speed + route.phase) % 1;
      const redProgress = (t * (route.speed * 0.85) + route.phase + 0.5) % 1;

      if (bluePacket) {
        const bluePos = route.curve.getPoint(blueProgress);
        bluePacket.position.set(bluePos.x, bluePos.y, bluePos.z);
      }

      if (redPacket) {
        const redPos = route.curve.getPoint(redProgress);
        redPacket.position.set(redPos.x, redPos.y, redPos.z);
      }

      const blueTrail = blueTrailRefs.current[idx] || [];
      for (let trailIdx = 0; trailIdx < blueTrail.length; trailIdx += 1) {
        const trailMesh = blueTrail[trailIdx];
        if (!trailMesh) continue;
        const trailProgress = wrap01(blueProgress - (trailIdx + 1) * 0.03);
        const trailPos = route.curve.getPoint(trailProgress);
        trailMesh.position.set(trailPos.x, trailPos.y, trailPos.z);
      }

      const redTrail = redTrailRefs.current[idx] || [];
      for (let trailIdx = 0; trailIdx < redTrail.length; trailIdx += 1) {
        const trailMesh = redTrail[trailIdx];
        if (!trailMesh) continue;
        const trailProgress = wrap01(redProgress - (trailIdx + 1) * 0.028);
        const trailPos = route.curve.getPoint(trailProgress);
        trailMesh.position.set(trailPos.x, trailPos.y, trailPos.z);
      }
    });
  });

  return (
    <group>
      {routes.map((route, idx) => {
        const points = route.curve.getPoints(48);
        const routeColor = idx % 3 === 0 ? ROUTE_COLOR_RED : ROUTE_COLOR_BLUE;
        return (
          <React.Fragment key={idx}>
            <Line points={points} color={routeColor} transparent opacity={0.26} lineWidth={1} />
            <mesh ref={(el) => { packetBlueRefs.current[idx] = el; }}>
              <sphereGeometry args={[0.0082, 10, 10]} />
              <meshBasicMaterial
                color={PACKET_COLOR_BLUE}
                transparent
                opacity={0.96}
                blending={THREE.AdditiveBlending}
                depthWrite={false}
              />
            </mesh>
            <mesh ref={(el) => { packetRedRefs.current[idx] = el; }}>
              <sphereGeometry args={[0.0078, 10, 10]} />
              <meshBasicMaterial
                color={PACKET_COLOR_RED}
                transparent
                opacity={0.95}
                blending={THREE.AdditiveBlending}
                depthWrite={false}
              />
            </mesh>
            {Array.from({ length: TRAIL_SEGMENTS }).map((_, trailIdx) => (
              <mesh
                key={`blue-trail-${idx}-${trailIdx}`}
                ref={(el) => {
                  if (!blueTrailRefs.current[idx]) blueTrailRefs.current[idx] = [];
                  blueTrailRefs.current[idx][trailIdx] = el;
                }}
              >
                <sphereGeometry args={[0.0068 - trailIdx * 0.0008, 8, 8]} />
                <meshBasicMaterial
                  color={PACKET_COLOR_BLUE}
                  transparent
                  opacity={0.5 - trailIdx * 0.035}
                  blending={THREE.AdditiveBlending}
                  depthWrite={false}
                />
              </mesh>
            ))}
            {Array.from({ length: TRAIL_SEGMENTS }).map((_, trailIdx) => (
              <mesh
                key={`red-trail-${idx}-${trailIdx}`}
                ref={(el) => {
                  if (!redTrailRefs.current[idx]) redTrailRefs.current[idx] = [];
                  redTrailRefs.current[idx][trailIdx] = el;
                }}
              >
                <sphereGeometry args={[0.0062 - trailIdx * 0.0007, 8, 8]} />
                <meshBasicMaterial
                  color={PACKET_COLOR_RED}
                  transparent
                  opacity={0.5 - trailIdx * 0.033}
                  blending={THREE.AdditiveBlending}
                  depthWrite={false}
                />
              </mesh>
            ))}
          </React.Fragment>
        );
      })}
    </group>
  );
}

function RotatingGlobeGroup() {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((_state, delta: number) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += AUTO_ROTATE_SPEED * delta;
    }
  });

  return (
    <group ref={groupRef}>
      <GlobePoints />
      <DataPacketRoutes />
    </group>
  );
}

function Scene() {
  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <RotatingGlobeGroup />
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
          dpr={[1, 1.5]}
        >
          <Scene />
        </Canvas>
      </Box>
    </Box>
  );
}
