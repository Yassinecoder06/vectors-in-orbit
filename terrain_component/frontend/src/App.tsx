import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars, Billboard, useTexture, Html } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import KeyboardControls from "./KeyboardControls";
import CameraFocus from "./CameraFocus";
import ControlIndicators from "./ControlIndicators";
import TourPanel from "./TourPanel";
import {
  Streamlit,
  withStreamlitConnection,
  type ComponentProps,
} from "streamlit-component-lib";
import type { TerrainPayload, TerrainPoint, TerrainBounds } from "./types";

const mulberry32 = (seed: number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
};

const generateProceduralPayload = (seed = 1337): TerrainPayload => {
  const rand = mulberry32(seed);
  const points: TerrainPoint[] = [];
  const grid = 10;
  const spread = 18;
  const minPrice = 80;
  const maxPrice = 1800;
  const priceRange = maxPrice - minPrice;

  for (let x = -grid; x <= grid; x++) {
    for (let z = -grid; z <= grid; z++) {
      const nx = (x / grid) * Math.PI;
      const nz = (z / grid) * Math.PI;
      const wave = Math.sin(nx * 1.2 + seed * 0.07) + Math.cos(nz * 1.5 - seed * 0.11);
      const swirl = Math.sin((nx + nz) * 0.6 + seed * 0.04);
      const price = minPrice + rand() * priceRange;
      const priceNorm = (price - minPrice) / priceRange;
      const height = (priceNorm + 0.08 * (wave + swirl)) * 18;
      const colors = ["#2ecc71", "#f39c12", "#e74c3c"];
      points.push({
        id: `demo-${x}-${z}`,
        position: [x * (spread / grid), height, z * (spread / grid)],
        height,
        price,
        price_normalized: priceNorm,
        color: colors[Math.floor(rand() * colors.length)],
        brand: "Demo Brand",
        category: "Demo Category",
        name: `Sample ${Math.abs(x) + Math.abs(z)}`,
        score: rand(),
        risk_tolerance: 0.5,
      });
    }
  }

  const highlights = points
    .filter((_, idx) => idx % 25 === 0)
    .slice(0, 7)
    .map((point, index) => ({
      id: point.id,
      label: `#${index + 1} ${point.name}`,
      position: point.position,
      price: point.price,
      brand: point.brand,
      category: point.category,
      score: point.score,
    }));

  const xs = points.map((p) => p.position[0]);
  const zs = points.map((p) => p.position[2]);

  return {
    points,
    highlights,
    meta: {
      mode: "procedural",
      seed,
      bounds: {
        minX: Math.min(...xs),
        maxX: Math.max(...xs),
        minZ: Math.min(...zs),
        maxZ: Math.max(...zs),
      },
      price_range: { min: minPrice, max: maxPrice },
      height_scale: 18,
    },
  };
};

const truncateLabel = (value: string, maxLength = 22) => {
  if (!value) {
    return "";
  }
  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}…` : value;
};

const DEFAULT_BOUNDS: TerrainBounds = { minX: -20, maxX: 20, minZ: -20, maxZ: 20 };
const TERRAIN_RESOLUTION = 140;

const deriveBounds = (payload: TerrainPayload): TerrainBounds => {
  if (payload.meta?.bounds) {
    return payload.meta.bounds;
  }
  if (!payload.points.length) {
    return DEFAULT_BOUNDS;
  }
  const xs = payload.points.map((p) => p.position[0]);
  const zs = payload.points.map((p) => p.position[2]);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minZ: Math.min(...zs),
    maxZ: Math.max(...zs),
  };
};

const sampleHeightAt = (x: number, z: number, points: TerrainPoint[], sigma: number): number => {
  if (!points.length) {
    return 0;
  }
  const sigma2 = sigma * sigma;
  let weightedSum = 0;
  let totalWeight = 0;
  for (const point of points) {
    const dx = x - point.position[0];
    const dz = z - point.position[2];
    const dist2 = dx * dx + dz * dz;
    if (dist2 < 1e-6) {
      return point.height;
    }
    const influence = Math.exp(-dist2 / (2 * sigma2));
    weightedSum += point.height * influence;
    totalWeight += influence;
  }
  return totalWeight > 1e-6 ? weightedSum / totalWeight : 0;
};

const mountainNoise = (x: number, z: number, seed: number) => {
  const f1 = 0.025;
  const f2 = 0.055;
  const f3 = 0.011;
  const ridge = Math.sin(x * f1 + seed * 0.01) * 0.8;
  const trough = Math.cos(z * (f1 * 1.8) - seed * 0.017) * 0.6;
  const swirl = Math.sin((x + z) * f2 + seed * 0.006) * 0.45;
  const basin = Math.cos((x - z) * f3 + seed * 0.014) * 0.3;
  return ridge + trough + swirl + basin;
};

const buildTerrainGeometry = (
  points: TerrainPoint[],
  bounds: TerrainBounds,
  seed: number
): THREE.BufferGeometry | null => {
  if (!points.length) {
    return null;
  }

  const width = Math.max(bounds.maxX - bounds.minX, 1);
  const depth = Math.max(bounds.maxZ - bounds.minZ, 1);
  const resolution = TERRAIN_RESOLUTION;
  const vertices = new Float32Array((resolution + 1) * (resolution + 1) * 3);
  const colors = new Float32Array((resolution + 1) * (resolution + 1) * 3);
  const heights = new Float32Array((resolution + 1) * (resolution + 1));
  const indices = new Uint32Array(resolution * resolution * 6);
  const stepX = width / resolution;
  const stepZ = depth / resolution;
  const smoothing = Math.max(width, depth) * 0.18;
  const noiseAmplitude = Math.max(width, depth) * 0.15;

  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (let iz = 0; iz <= resolution; iz++) {
    for (let ix = 0; ix <= resolution; ix++) {
      const vertIndex = iz * (resolution + 1) + ix;
      const x = bounds.minX + ix * stepX;
      const z = bounds.minZ + iz * stepZ;
      const baseHeight = sampleHeightAt(x, z, points, smoothing);
      const noise = mountainNoise(x, z, seed) * noiseAmplitude;
      const y = baseHeight + noise;
      heights[vertIndex] = y;
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      vertices[vertIndex * 3 + 0] = x;
      vertices[vertIndex * 3 + 1] = y;
      vertices[vertIndex * 3 + 2] = z;
    }
  }

  const rangeY = Math.max(maxY - minY, 1e-6);
  const color = new THREE.Color();
  for (let idx = 0; idx < heights.length; idx++) {
    const t = (heights[idx] - minY) / rangeY;
    const eased = Math.pow(t, 1.2);
    color.setHSL(0.55 - 0.35 * eased, 0.55 + 0.15 * (1 - eased), 0.3 + 0.4 * eased);
    colors[idx * 3 + 0] = color.r;
    colors[idx * 3 + 1] = color.g;
    colors[idx * 3 + 2] = color.b;
  }

  let idx = 0;
  for (let iz = 0; iz < resolution; iz++) {
    for (let ix = 0; ix < resolution; ix++) {
      const a = iz * (resolution + 1) + ix;
      const b = a + 1;
      const c = a + (resolution + 1);
      const d = c + 1;
      indices[idx++] = a;
      indices[idx++] = c;
      indices[idx++] = b;
      indices[idx++] = c;
      indices[idx++] = d;
      indices[idx++] = b;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  return geometry;
};

const TerrainSurface = ({
  points,
  bounds,
  seed,
}: {
  points: TerrainPoint[];
  bounds: TerrainBounds;
  seed: number;
}) => {
  const geometry = useMemo(() => buildTerrainGeometry(points, bounds, seed), [points, bounds, seed]);

  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  if (!geometry) {
    return null;
  }

  return (
    <mesh geometry={geometry} receiveShadow castShadow>
      <meshStandardMaterial vertexColors roughness={0.85} metalness={0.2} />
    </mesh>
  );
};

const ProductMarkers = ({
  points,
  onSelect,
}: {
  points: TerrainPoint[];
  onSelect: (point: TerrainPoint) => void;
}) => (
  <group>
    {points.map((point) => (
      <mesh
        key={point.id}
        position={[point.position[0], point.height + 0.2, point.position[2]]}
        onClick={(event) => {
          event.stopPropagation();
          onSelect(point);
        }}
      >
        <sphereGeometry args={[0.18, 18, 18]} />
        <meshStandardMaterial color={point.color} emissive={point.color} emissiveIntensity={0.15} />
      </mesh>
    ))}
  </group>
);

const ProductBillboard = ({
  point,
  onSelect,
}: {
  point: TerrainPoint;
  onSelect: (point: TerrainPoint) => void;
}) => {
  const texture = useTexture(point.imageUrl!, (tex) => {
    tex.anisotropy = 8;
  });

  const size = THREE.MathUtils.clamp(1.6 + (point.price_normalized ?? 0) * 1.8, 1.5, 3.2) * 3.5;
  const baseLift = size * 0.75 + 0.6;
  const imageOffset = size * 0.25;
  return (
    <Billboard
      position={[point.position[0], point.height + baseLift, point.position[2]]}
      follow
    >
      <mesh
        position={[0, imageOffset, 0]}
        onClick={(event) => {
          event.stopPropagation();
          onSelect(point);
        }}
      >
        <planeGeometry args={[size, size * 0.75]} />
        <meshBasicMaterial map={texture} transparent side={THREE.DoubleSide} depthTest={false} />
      </mesh>
    </Billboard>
  );
};

const ProductBillboards = ({
  points,
  onSelect,
}: {
  points: TerrainPoint[];
  onSelect: (point: TerrainPoint) => void;
}) => (
  <group>
    {points
      .filter((p) => !!p.imageUrl)
      .map((p) => (
        <ProductBillboard key={`billboard-${p.id}`} point={p} onSelect={onSelect} />
      ))}
  </group>
);

const ProductLabels = ({
  points,
  onSelect,
}: {
  points: TerrainPoint[];
  onSelect: (point: TerrainPoint) => void;
}) => (
  <group>
    {points.map((point) => (
      <Billboard
        key={`label-${point.id}`}
        position={[point.position[0], point.height + 0.9, point.position[2]]}
        follow
      >
        <Html center transform distanceFactor={8} style={{ pointerEvents: "auto" }}>
          <div
            className="product-label"
            onClick={(event) => {
              event.stopPropagation();
              onSelect(point);
            }}
          >
            {truncateLabel(point.name)}
          </div>
        </Html>
      </Billboard>
    ))}
  </group>
);

const HighlightMarkers = ({ points }: { points: TerrainPoint[] }) => (
  <group>
    {points.map((point) => (
      <mesh key={`marker-${point.id}`} position={[point.position[0], point.height + 0.6, point.position[2]]}>
        <sphereGeometry args={[0.28, 32, 32]} />
        <meshStandardMaterial color="#f8fafc" emissive="#fde047" emissiveIntensity={0.8} />
      </mesh>
    ))}
  </group>
);

const NarrationPanel = ({ payload }: { payload: TerrainPayload }) => {
  if (!payload.highlights.length) {
    return null;
  }
  return (
    <div className="narration-panel">
      <h3>Guide Picks</h3>
      {payload.highlights.map((entry) => (
        <div className="highlight-row" key={entry.id}>
          <span>{entry.label}</span>
          <span>
            ${" "}
            {entry.price.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        </div>
      ))}
    </div>
  );
};

function Scene({
  payload,
  onSelect,
  focusPosition,
  onFocusComplete,
}: {
  payload: TerrainPayload;
  onSelect: (point: TerrainPoint) => void;
  focusPosition: [number, number, number] | null;
  onFocusComplete?: () => void;
}) {
  const bounds = useMemo(() => deriveBounds(payload), [payload]);
  const highlightPoints = useMemo(
    () => payload.highlights.map((highlight) => payload.points.find((p) => p.id === highlight.id)).filter(Boolean) as TerrainPoint[],
    [payload]
  );
  const meshSeed = payload.meta?.seed ?? 1337;

  // Ref for OrbitControls to enable keyboard navigation
  const orbitControlsRef = useRef<{
    target: THREE.Vector3;
    update: () => void;
    object: THREE.Camera;
  } | null>(null);

  const width = Math.max(bounds.maxX - bounds.minX, 1);
  const depth = Math.max(bounds.maxZ - bounds.minZ, 1);
  const diag = Math.max(width, depth);
  const orbitTarget: [number, number, number] = [
    (bounds.minX + bounds.maxX) / 2,
    0,
    (bounds.minZ + bounds.maxZ) / 2,
  ];
  const cameraPosition: [number, number, number] = [
    bounds.maxX + diag * 0.6,
    diag + 300,
    bounds.maxZ + diag * 0.6,
  ];

  return (
    <Canvas shadows camera={{ position: cameraPosition, fov: 38 }}>
      <color attach="background" args={["#030712"]} />
      <ambientLight intensity={0.4} />
      <directionalLight position={[25, 40, 10]} intensity={1.2} castShadow />
      <pointLight position={[-20, 15, -10]} intensity={0.5} />
      <Stars radius={80} depth={40} factor={4} fade speed={0.4} />
      <TerrainSurface points={payload.points} bounds={bounds} seed={meshSeed} />
      <ProductMarkers points={payload.points} onSelect={onSelect} />
      <ProductBillboards points={payload.points} onSelect={onSelect} />
      <ProductLabels points={payload.points} onSelect={onSelect} />
      <HighlightMarkers points={highlightPoints} />
      <OrbitControls
        ref={orbitControlsRef as any}
        target={orbitTarget}
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        maxDistance={diag * 4}
        minDistance={diag * 0.3}
        maxPolarAngle={Math.PI / 2.05}
        enableDamping
        dampingFactor={0.05}
      />
      {/* WASD keyboard navigation */}
      <KeyboardControls
        enabled={true}
        moveSpeed={diag * 0.02}
        orbitControlsRef={orbitControlsRef}
      />
      {/* Smooth camera focus animation */}
      <CameraFocus
        focusPosition={focusPosition}
        orbitControlsRef={orbitControlsRef}
        onFocusComplete={onFocusComplete}
      />
    </Canvas>
  );
}

const TerrainApp = (props: ComponentProps) => {
  const rawPayload = (props.args?.["data"] as TerrainPayload | undefined) ?? undefined;
  const payload = useMemo<TerrainPayload>(() => {
    if (rawPayload?.points?.length) {
      return rawPayload;
    }
    return generateProceduralPayload();
  }, [rawPayload]);

  // Tour state
  const [tourActive, setTourActive] = useState(false);
  const [tourStep, setTourStep] = useState(0); // 0 = intro, 1-N = products, N+1 = outro
  const [focusPosition, setFocusPosition] = useState<[number, number, number] | null>(null);

  // Get tour products from highlights
  const tourProducts = useMemo(() => {
    return payload.highlights
      .map((h) => payload.points.find((p) => p.id === h.id))
      .filter(Boolean) as TerrainPoint[];
  }, [payload]);

  const totalSteps = tourProducts.length;

  // Get current tour product
  const currentTourProduct = useMemo(() => {
    if (!tourActive || tourStep < 1 || tourStep > totalSteps) return null;
    return tourProducts[tourStep - 1] || null;
  }, [tourActive, tourStep, tourProducts, totalSteps]);

  // Focus camera on current tour product
  useEffect(() => {
    if (currentTourProduct) {
      setFocusPosition([
        currentTourProduct.position[0],
        currentTourProduct.height,
        currentTourProduct.position[2],
      ]);
    }
  }, [currentTourProduct]);

  // Tour navigation handlers
  const handleTourStart = useCallback(() => {
    setTourActive(true);
    setTourStep(0);
  }, []);

  const handleTourEnd = useCallback(() => {
    setTourActive(false);
    setTourStep(0);
    setFocusPosition(null);
  }, []);

  const handleTourNext = useCallback(() => {
    if (tourStep > totalSteps) {
      handleTourEnd();
    } else {
      setTourStep((prev) => prev + 1);
    }
  }, [tourStep, totalSteps, handleTourEnd]);

  const handleTourPrevious = useCallback(() => {
    setTourStep((prev) => Math.max(0, prev - 1));
  }, []);

  // Handle keyboard navigation for tour
  useEffect(() => {
    if (!tourActive) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === 'n') {
        handleTourNext();
      } else if (e.key === 'ArrowLeft' || e.key === 'p') {
        handleTourPrevious();
      } else if (e.key === 'Escape') {
        handleTourEnd();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [tourActive, handleTourNext, handleTourPrevious, handleTourEnd]);

  const handleSelect = (point: TerrainPoint) => {
    // Focus camera on selected point
    setFocusPosition([point.position[0], point.height, point.position[2]]);
    
    Streamlit.setComponentValue({
      id: point.id,
      name: point.name,
      description: point.description ?? "",
      price: point.price,
      brand: point.brand,
      category: point.category,
      imageUrl: point.imageUrl,
      score: point.score,
      risk_tolerance: point.risk_tolerance,
    });
  };

  const handleFocusComplete = useCallback(() => {
    // Optional: auto-clear focus after animation
  }, []);

  useEffect(() => {
    Streamlit.setFrameHeight();
  }, [payload, props.width]);

  return (
    <div className="terrain-shell">
      <Scene 
        payload={payload} 
        onSelect={handleSelect}
        focusPosition={focusPosition}
        onFocusComplete={handleFocusComplete}
      />
      <NarrationPanel payload={payload} />
      <ControlIndicators visible={!tourActive} />
      <TourPanel
        tourActive={tourActive}
        currentStep={tourStep}
        totalSteps={totalSteps}
        currentProduct={currentTourProduct}
        onNext={handleTourNext}
        onPrevious={handleTourPrevious}
        onStart={handleTourStart}
        onEnd={handleTourEnd}
        highlights={payload.highlights}
      />
    </div>
  );
};

export default withStreamlitConnection(TerrainApp);
