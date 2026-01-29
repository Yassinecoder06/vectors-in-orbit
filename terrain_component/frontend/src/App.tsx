import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars, Billboard, useTexture, Html, Line, Cloud, Sky } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import KeyboardControls from "./KeyboardControls";
import CameraFocus from "./CameraFocus";
import ControlIndicators from "./ControlIndicators";
import TourPanel from "./TourPanel";
import QueryPath from "./QueryPath";
import {
  generateMountainsFromProducts,
  generateRiverPath,
  buildHeightGrid,
  buildTerrainMesh,
  sampleTerrainHeight,
  type Mountain,
  type RiverPoint,
} from "./terrainUtils";
import {
  Streamlit,
  withStreamlitConnection,
  type ComponentProps,
} from "streamlit-component-lib";
import type { TerrainPayload, TerrainPoint, TerrainBounds } from "./types";

// Fluffy cartoon cloud component
const CartoonCloud = ({ 
  position, 
  scale = 1 
}: { 
  position: [number, number, number]; 
  scale?: number;
}) => {
  const cloudColor = "#FFFFFF";
  return (
    <group position={position} scale={scale}>
      {/* Main body */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[3, 16, 16]} />
        <meshBasicMaterial color={cloudColor} />
      </mesh>
      {/* Left bump */}
      <mesh position={[-2.5, 0.5, 0]}>
        <sphereGeometry args={[2.2, 16, 16]} />
        <meshBasicMaterial color={cloudColor} />
      </mesh>
      {/* Right bump */}
      <mesh position={[2.5, 0.3, 0]}>
        <sphereGeometry args={[2.5, 16, 16]} />
        <meshBasicMaterial color={cloudColor} />
      </mesh>
      {/* Top bump */}
      <mesh position={[0.5, 1.8, 0]}>
        <sphereGeometry args={[2, 16, 16]} />
        <meshBasicMaterial color={cloudColor} />
      </mesh>
      {/* Extra bumps for fluffiness */}
      <mesh position={[-1.5, 1.2, 0.5]}>
        <sphereGeometry args={[1.5, 12, 12]} />
        <meshBasicMaterial color={cloudColor} />
      </mesh>
      <mesh position={[1.8, 1, -0.3]}>
        <sphereGeometry args={[1.8, 12, 12]} />
        <meshBasicMaterial color={cloudColor} />
      </mesh>
    </group>
  );
};

// Sky decoration with clouds
const SkyDecoration = ({ bounds, seed }: { bounds: TerrainBounds; seed: number }) => {
  const clouds = useMemo(() => {
    const rand = mulberry32(seed + 777);
    const cloudList: Array<{
      id: string;
      position: [number, number, number];
      scale: number;
    }> = [];
    
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerZ = (bounds.minZ + bounds.maxZ) / 2;
    const spread = Math.max(bounds.maxX - bounds.minX, bounds.maxZ - bounds.minZ) * 1.2;
    
    // Generate 8-12 clouds
    const numClouds = 8 + Math.floor(rand() * 5);
    
    for (let i = 0; i < numClouds; i++) {
      const angle = (i / numClouds) * Math.PI * 2 + rand() * 0.5;
      const dist = spread * 0.4 + rand() * spread * 0.4;
      
      cloudList.push({
        id: `cloud-${i}`,
        position: [
          centerX + Math.cos(angle) * dist,
          35 + rand() * 25,
          centerZ + Math.sin(angle) * dist,
        ],
        scale: 0.8 + rand() * 0.6,
      });
    }
    
    return cloudList;
  }, [bounds, seed]);
  
  return (
    <group>
      {clouds.map((cloud) => (
        <CartoonCloud
          key={cloud.id}
          position={cloud.position}
          scale={cloud.scale}
        />
      ))}
    </group>
  );
};

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
const TERRAIN_RESOLUTION = 160;

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

// River component - renders the water surface
const RiverMesh = ({ riverPoints, bounds }: { riverPoints: RiverPoint[]; bounds: TerrainBounds }) => {
  if (riverPoints.length < 2) return null;
  
  const riverGeometry = useMemo(() => {
    const riverWidth = 4;
    const vertices: number[] = [];
    const indices: number[] = [];
    
    for (let i = 0; i < riverPoints.length - 1; i++) {
      const p1 = riverPoints[i];
      const p2 = riverPoints[i + 1];
      
      // Direction along river
      const dx = p2.x - p1.x;
      const dz = p2.z - p1.z;
      const len = Math.sqrt(dx * dx + dz * dz);
      
      // Perpendicular for width
      const nx = -dz / len * riverWidth / 2;
      const nz = dx / len * riverWidth / 2;
      
      // Four corners of this river segment
      const idx = vertices.length / 3;
      
      // Left side of start
      vertices.push(p1.x + nx, -0.8, p1.z + nz);
      // Right side of start
      vertices.push(p1.x - nx, -0.8, p1.z - nz);
      // Left side of end
      vertices.push(p2.x + nx, -0.8, p2.z + nz);
      // Right side of end
      vertices.push(p2.x - nx, -0.8, p2.z - nz);
      
      // Two triangles
      indices.push(idx, idx + 2, idx + 1);
      indices.push(idx + 1, idx + 2, idx + 3);
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    return geometry;
  }, [riverPoints]);
  
  return (
    <mesh geometry={riverGeometry}>
      <meshStandardMaterial
        color="#5DADE2"
        transparent
        opacity={0.85}
        metalness={0.4}
        roughness={0.2}
      />
    </mesh>
  );
};

// Main terrain surface using the new terrain generation system
const TerrainSurface = ({
  points,
  bounds,
  seed,
}: {
  points: TerrainPoint[];
  bounds: TerrainBounds;
  seed: number;
}) => {
  const { geometry, riverPoints } = useMemo(() => {
    const width = Math.max(bounds.maxX - bounds.minX, 80);
    const depth = Math.max(bounds.maxZ - bounds.minZ, 80);
    const terrainSize = Math.max(width, depth) * 1.3;
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerZ = (bounds.minZ + bounds.maxZ) / 2;
    
    // Generate mountains from product positions
    const { mountains, hills } = generateMountainsFromProducts(points, bounds);
    
    // Generate river through low areas
    const riverPoints = generateRiverPath(mountains, {
      minX: centerX - terrainSize / 2,
      maxX: centerX + terrainSize / 2,
      minZ: centerZ - terrainSize / 2,
      maxZ: centerZ + terrainSize / 2,
    });
    
    // Build terrain config
    const config = {
      width: terrainSize,
      height: terrainSize,
      resolution: TERRAIN_RESOLUTION,
      mountains,
      hills,
      riverPoints,
      seed,
      circularBoundary: true,
      boundaryRadius: terrainSize / 2 * 0.95,
    };
    
    // Generate height grid and mesh
    const heightGrid = buildHeightGrid(config);
    const geometry = buildTerrainMesh(heightGrid, config);
    
    // Offset geometry to terrain center
    geometry.translate(centerX, 0, centerZ);
    
    return { geometry, riverPoints: riverPoints.map(p => ({ x: p.x + centerX, z: p.z + centerZ })) };
  }, [points, bounds, seed]);

  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  return (
    <group>
      <mesh geometry={geometry} receiveShadow castShadow>
        <meshStandardMaterial vertexColors roughness={0.8} metalness={0.15} />
      </mesh>
      <RiverMesh riverPoints={riverPoints} bounds={bounds} />
    </group>
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

// Simple low-poly tree
const LowPolyTree = ({ 
  position, 
  scale = 1, 
  color = "#228B22" 
}: { 
  position: [number, number, number]; 
  scale?: number; 
  color?: string;
}) => (
  <group position={position} scale={scale}>
    {/* Trunk */}
    <mesh position={[0, 0.4, 0]}>
      <cylinderGeometry args={[0.1, 0.15, 0.8, 6]} />
      <meshStandardMaterial color="#8B4513" />
    </mesh>
    {/* Foliage - stacked cones */}
    <mesh position={[0, 1.2, 0]}>
      <coneGeometry args={[0.6, 1.2, 6]} />
      <meshStandardMaterial color={color} />
    </mesh>
    <mesh position={[0, 1.8, 0]}>
      <coneGeometry args={[0.45, 0.9, 6]} />
      <meshStandardMaterial color={color} />
    </mesh>
    <mesh position={[0, 2.3, 0]}>
      <coneGeometry args={[0.3, 0.6, 6]} />
      <meshStandardMaterial color={color} />
    </mesh>
  </group>
);

// Forest decoration - places trees around the terrain
const ForestDecoration = ({ bounds, seed }: { bounds: TerrainBounds; seed: number }) => {
  const trees = useMemo(() => {
    const treeList: Array<{
      id: string;
      position: [number, number, number];
      scale: number;
      color: string;
    }> = [];
    
    const rand = mulberry32(seed + 999);
    const width = bounds.maxX - bounds.minX;
    const depth = bounds.maxZ - bounds.minZ;
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerZ = (bounds.minZ + bounds.maxZ) / 2;
    const radius = Math.max(width, depth) / 2 * 0.85;
    
    const treeColors = [
      "#228B22", // Forest green
      "#2E8B57", // Sea green
      "#006400", // Dark green
      "#32CD32", // Lime green
      "#FF8C00", // Autumn orange
      "#DC143C", // Autumn red
    ];
    
    // Generate 40-60 trees
    const numTrees = 40 + Math.floor(rand() * 20);
    
    for (let i = 0; i < numTrees; i++) {
      // Random position within terrain bounds (polar coordinates for circular distribution)
      const angle = rand() * Math.PI * 2;
      const dist = (0.2 + rand() * 0.7) * radius; // 20-90% of radius
      
      const x = centerX + Math.cos(angle) * dist;
      const z = centerZ + Math.sin(angle) * dist;
      
      // Height at terrain surface (approximation - in practice would sample terrain)
      // For now, use a simple formula
      const distFromCenter = Math.sqrt((x - centerX) ** 2 + (z - centerZ) ** 2);
      const normalizedDist = distFromCenter / radius;
      
      // Trees on lower slopes (avoid peaks and water)
      if (normalizedDist > 0.15 && normalizedDist < 0.8) {
        const baseHeight = (1 - normalizedDist) * 8 + rand() * 2;
        
        // Skip if would be in water
        if (baseHeight > 0.5) {
          const scale = 0.6 + rand() * 0.8;
          const colorIdx = Math.floor(rand() * treeColors.length);
          
          treeList.push({
            id: `tree-${i}`,
            position: [x, baseHeight, z],
            scale,
            color: treeColors[colorIdx],
          });
        }
      }
    }
    
    return treeList;
  }, [bounds, seed]);
  
  return (
    <group>
      {trees.map((tree) => (
        <LowPolyTree
          key={tree.id}
          position={tree.position}
          scale={tree.scale}
          color={tree.color}
        />
      ))}
    </group>
  );
};

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
  showQueryPath,
}: {
  payload: TerrainPayload;
  onSelect: (point: TerrainPoint) => void;
  focusPosition: [number, number, number] | null;
  onFocusComplete?: () => void;
  showQueryPath?: boolean;
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
    8,  // Slightly elevated target for better mountain view
    (bounds.minZ + bounds.maxZ) / 2,
  ];
  // Camera positioned further back and at better angle for full mountain view
  const cameraPosition: [number, number, number] = [
    bounds.maxX + diag * 0.4,
    diag * 0.5 + 20,
    bounds.maxZ + diag * 0.4,
  ];

  return (
    <Canvas shadows camera={{ position: cameraPosition, fov: 38 }}>
      <color attach="background" args={["#87CEEB"]} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[40, 60, 30]} intensity={1.4} castShadow />
      <pointLight position={[-30, 25, -20]} intensity={0.4} color="#fff5e6" />
      <hemisphereLight color="#87CEEB" groundColor="#228B22" intensity={0.3} />
      <SkyDecoration bounds={bounds} seed={meshSeed} />
      <TerrainSurface points={payload.points} bounds={bounds} seed={meshSeed} />
      <ForestDecoration bounds={bounds} seed={meshSeed} />
      {/* Query path showing the journey through ranked products */}
      <QueryPath points={payload.points} visible={showQueryPath ?? true} />
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
        maxDistance={diag * 5}
        minDistance={diag * 0.25}
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
        showQueryPath={true}
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
