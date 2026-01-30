import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars, Billboard, useTexture, Html, Line, Cloud, Sky } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import KeyboardControls from "./KeyboardControls";
import CameraFocus from "./CameraFocus";
import ControlIndicators from "./ControlIndicators";
import TourPanel from "./TourPanel";
import QueryPath from "./QueryPath";
import FilterPanel, { FilterState, getDefaultFilters, applyFilters } from "./FilterPanel";
import ComparisonPanel from "./ComparisonPanel";
import ProductPreview from "./ProductPreview";
import SearchPanel from "./SearchPanel";
import ScoreBreakdown from "./ScoreBreakdown";
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

// Height sampling function type
type HeightSampler = (x: number, z: number) => number;

// Main terrain surface using the new terrain generation system
// Returns height sampler function for products to use
const useTerrainData = (
  points: TerrainPoint[],
  bounds: TerrainBounds,
  seed: number
) => {
  return useMemo(() => {
    const width = Math.max(bounds.maxX - bounds.minX, 80);
    const depth = Math.max(bounds.maxZ - bounds.minZ, 80);
    const terrainSize = Math.max(width, depth) * 1.3;
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerZ = (bounds.minZ + bounds.maxZ) / 2;
    
    // Generate mountains from product positions
    const { mountains, hills } = generateMountainsFromProducts(points, bounds);
    
    // Generate river around the mountain
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
    
    // Create height sampler function that samples the actual terrain
    const sampleHeight: HeightSampler = (worldX: number, worldZ: number) => {
      // Convert world coords to local terrain coords
      const localX = worldX - centerX;
      const localZ = worldZ - centerZ;
      return sampleTerrainHeight(heightGrid, localX, localZ, terrainSize, terrainSize, TERRAIN_RESOLUTION);
    };
    
    // Adjust river points to world coordinates
    const worldRiverPoints = riverPoints.map(p => ({ x: p.x + centerX, z: p.z + centerZ }));
    
    return { 
      geometry, 
      riverPoints: worldRiverPoints, 
      sampleHeight,
      terrainSize,
      centerX,
      centerZ,
    };
  }, [points, bounds, seed]);
};

// Main terrain surface component with texture
const TerrainSurface = ({
  geometry,
  riverPoints,
  bounds,
}: {
  geometry: THREE.BufferGeometry;
  riverPoints: RiverPoint[];
  bounds: TerrainBounds;
}) => {
  // Load terrain texture
  const terrainTexture = useTexture('/diffuse_1SG_baseColor.jpeg', (tex) => {
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(8, 8); // Tile the texture across the terrain
    tex.anisotropy = 16;
  });

  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  return (
    <group>
      <mesh geometry={geometry} receiveShadow castShadow>
        <meshStandardMaterial 
          vertexColors 
          map={terrainTexture}
          roughness={0.85} 
          metalness={0.1}
        />
      </mesh>
      <RiverMesh riverPoints={riverPoints} bounds={bounds} />
    </group>
  );
};

const ProductMarkers = ({
  points,
  onSelect,
  sampleHeight,
}: {
  points: TerrainPoint[];
  onSelect: (point: TerrainPoint) => void;
  sampleHeight: HeightSampler;
}) => (
  <group>
    {points.map((point) => {
      // Sample actual terrain height at product position
      const terrainHeight = sampleHeight(point.position[0], point.position[2]);
      const y = Math.max(terrainHeight, point.height) + 0.3; // Use higher of terrain or product height
      return (
        <mesh
          key={point.id}
          position={[point.position[0], y, point.position[2]]}
          onClick={(event) => {
            event.stopPropagation();
            onSelect(point);
          }}
        >
          <sphereGeometry args={[0.18, 18, 18]} />
          <meshStandardMaterial color={point.color} emissive={point.color} emissiveIntensity={0.15} />
        </mesh>
      );
    })}
  </group>
);

const ProductBillboard = ({
  point,
  onSelect,
  sampleHeight,
}: {
  point: TerrainPoint;
  onSelect: (point: TerrainPoint) => void;
  sampleHeight: HeightSampler;
}) => {
  const texture = useTexture(point.imageUrl!, (tex) => {
    tex.anisotropy = 8;
  });

  // Sample actual terrain height
  const terrainHeight = sampleHeight(point.position[0], point.position[2]);
  const baseY = Math.max(terrainHeight, point.height);
  
  const size = THREE.MathUtils.clamp(1.6 + (point.price_normalized ?? 0) * 1.8, 1.5, 3.2) * 3.5;
  const baseLift = size * 0.75 + 0.6;
  const imageOffset = size * 0.25;
  return (
    <Billboard
      position={[point.position[0], baseY + baseLift, point.position[2]]}
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
  sampleHeight,
}: {
  points: TerrainPoint[];
  onSelect: (point: TerrainPoint) => void;
  sampleHeight: HeightSampler;
}) => (
  <group>
    {points
      .filter((p) => !!p.imageUrl)
      .map((p) => (
        <ProductBillboard key={`billboard-${p.id}`} point={p} onSelect={onSelect} sampleHeight={sampleHeight} />
      ))}
  </group>
);

const ProductLabels = ({
  points,
  onSelect,
  sampleHeight,
  onHover,
  highlightedIds,
  onRightClick,
}: {
  points: TerrainPoint[];
  onSelect: (point: TerrainPoint) => void;
  sampleHeight: HeightSampler;
  onHover?: (point: TerrainPoint | null, event?: React.MouseEvent) => void;
  highlightedIds?: Set<string>;
  onRightClick?: (point: TerrainPoint, event: React.MouseEvent) => void;
}) => (
  <group>
    {points.map((point, index) => {
      const terrainHeight = sampleHeight(point.position[0], point.position[2]);
      const y = Math.max(terrainHeight, point.height) + 1.2;
      const matchScore = Math.round((point.score ?? 0) * 100);
      const isHighlighted = highlightedIds?.has(point.id);
      const isTopProduct = index < 3; // Top 3 products get special treatment
      
      return (
        <Billboard
          key={`label-${point.id}`}
          position={[point.position[0], y, point.position[2]]}
          follow
        >
          <Html center transform distanceFactor={8} style={{ pointerEvents: "auto" }}>
            <div
              className={`product-label ${isHighlighted ? 'highlighted' : ''} ${isTopProduct ? 'top-product' : ''}`}
              data-rank={index + 1}
              style={{ animationDelay: `${index * 0.05}s` }}
              onClick={(event) => {
                event.stopPropagation();
                onSelect(point);
              }}
              onMouseEnter={(event) => onHover?.(point, event)}
              onMouseLeave={() => onHover?.(null)}
              onContextMenu={(event) => {
                event.preventDefault();
                event.stopPropagation();
                onRightClick?.(point, event);
              }}
            >
              {isTopProduct && <span className="rank-badge">#{index + 1}</span>}
              <span className="product-name">{truncateLabel(point.name)}</span>
              <span className="product-score">{matchScore}</span>
            </div>
          </Html>
        </Billboard>
      );
    })}
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

// Bush component for ground cover
const Bush = ({ position, scale = 1, color = "#2E8B57" }: { 
  position: [number, number, number]; 
  scale?: number;
  color?: string;
}) => (
  <group position={position} scale={scale}>
    <mesh position={[0, 0.2, 0]}>
      <sphereGeometry args={[0.3, 8, 8]} />
      <meshStandardMaterial color={color} />
    </mesh>
    <mesh position={[0.2, 0.15, 0.1]}>
      <sphereGeometry args={[0.2, 8, 8]} />
      <meshStandardMaterial color={color} />
    </mesh>
    <mesh position={[-0.15, 0.18, -0.1]}>
      <sphereGeometry args={[0.22, 8, 8]} />
      <meshStandardMaterial color={color} />
    </mesh>
  </group>
);

// Rock component for terrain detail
const Rock = ({ position, scale = 1 }: { 
  position: [number, number, number]; 
  scale?: number;
}) => (
  <group position={position} scale={scale}>
    <mesh rotation={[Math.random() * 0.5, Math.random() * Math.PI, 0]}>
      <dodecahedronGeometry args={[0.3, 0]} />
      <meshStandardMaterial color="#7f8c8d" roughness={0.9} />
    </mesh>
  </group>
);

// Cottage/House component for village feel
const Cottage = ({ 
  position, 
  scale = 1,
  roofColor = "#8B4513"
}: { 
  position: [number, number, number]; 
  scale?: number;
  roofColor?: string;
}) => (
  <group position={position} scale={scale}>
    {/* Base/walls */}
    <mesh position={[0, 0.4, 0]}>
      <boxGeometry args={[0.8, 0.8, 0.8]} />
      <meshStandardMaterial color="#F5DEB3" roughness={0.8} />
    </mesh>
    {/* Roof */}
    <mesh position={[0, 1, 0]} rotation={[0, Math.PI / 4, 0]}>
      <coneGeometry args={[0.7, 0.6, 4]} />
      <meshStandardMaterial color={roofColor} roughness={0.7} />
    </mesh>
    {/* Door */}
    <mesh position={[0, 0.25, 0.41]}>
      <boxGeometry args={[0.2, 0.4, 0.02]} />
      <meshStandardMaterial color="#654321" />
    </mesh>
    {/* Window */}
    <mesh position={[0.25, 0.5, 0.41]}>
      <boxGeometry args={[0.15, 0.15, 0.02]} />
      <meshStandardMaterial color="#87CEEB" metalness={0.3} />
    </mesh>
    {/* Chimney */}
    <mesh position={[0.25, 1.2, 0]}>
      <boxGeometry args={[0.15, 0.3, 0.15]} />
      <meshStandardMaterial color="#8B0000" roughness={0.9} />
    </mesh>
  </group>
);

// Sun component - glowing sphere in the sky
const Sun = ({ 
  position = [80, 60, -40] as [number, number, number],
  size = 12
}: { 
  position?: [number, number, number];
  size?: number;
}) => (
  <group position={position}>
    {/* Sun core - bright yellow */}
    <mesh>
      <sphereGeometry args={[size, 32, 32]} />
      <meshBasicMaterial color="#FDB813" />
    </mesh>
    {/* Sun glow - larger transparent sphere */}
    <mesh>
      <sphereGeometry args={[size * 1.3, 32, 32]} />
      <meshBasicMaterial color="#FFE484" transparent opacity={0.3} />
    </mesh>
    {/* Outer glow */}
    <mesh>
      <sphereGeometry args={[size * 1.8, 32, 32]} />
      <meshBasicMaterial color="#FFF5D6" transparent opacity={0.15} />
    </mesh>
    {/* Point light from sun */}
    <pointLight color="#FFEECC" intensity={0.8} distance={300} />
  </group>
);

// Ocean rim component - creates a visible water ring around the terrain
const OceanRim = ({ 
  bounds, 
  terrainRadius 
}: { 
  bounds: TerrainBounds; 
  terrainRadius: number;
}) => {
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerZ = (bounds.minZ + bounds.maxZ) / 2;
  
  return (
    <group position={[centerX, -1.5, centerZ]}>
      {/* Main ocean plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[terrainRadius * 0.85, terrainRadius * 1.5, 64]} />
        <meshStandardMaterial 
          color="#5DADE2" 
          transparent 
          opacity={0.85}
          metalness={0.3}
          roughness={0.2}
        />
      </mesh>
      {/* Deeper water further out */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.5, 0]}>
        <ringGeometry args={[terrainRadius * 1.2, terrainRadius * 2, 64]} />
        <meshStandardMaterial 
          color="#3498DB" 
          transparent 
          opacity={0.9}
          metalness={0.2}
          roughness={0.3}
        />
      </mesh>
    </group>
  );
};

// Forest decoration - places trees, bushes, rocks, and houses around the terrain
const ForestDecoration = ({ 
  bounds, 
  seed,
  sampleHeight 
}: { 
  bounds: TerrainBounds; 
  seed: number;
  sampleHeight?: HeightSampler;
}) => {
  const decorations = useMemo(() => {
    const treeList: Array<{
      id: string;
      position: [number, number, number];
      scale: number;
      color: string;
      type: 'tree' | 'bush' | 'rock' | 'house';
      rotation?: number;
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
      "#1a5f1a", // Deep forest
      "#3a9a3a", // Medium green
    ];
    
    const autumnColors = [
      "#FF8C00", // Autumn orange
      "#DC143C", // Autumn red
      "#DAA520", // Golden
      "#CD853F", // Peru
    ];
    
    const bushColors = [
      "#228B22",
      "#2E8B57", 
      "#3CB371",
      "#556B2F",
    ];
    
    const roofColors = [
      "#8B4513", // Saddle brown
      "#A0522D", // Sienna
      "#CD853F", // Peru
      "#8B0000", // Dark red
      "#654321", // Dark brown
    ];
    
    // Generate 200-260 trees (much denser forest)
    const numTrees = 200 + Math.floor(rand() * 60);
    
    // Generate trees in clusters for more natural look
    const numClusters = 12 + Math.floor(rand() * 5);
    const clusterCenters: Array<{ x: number; z: number; colorBias: number }> = [];
    
    for (let c = 0; c < numClusters; c++) {
      const angle = (c / numClusters) * Math.PI * 2 + rand() * 0.5;
      const dist = (0.2 + rand() * 0.55) * radius;
      clusterCenters.push({
        x: centerX + Math.cos(angle) * dist,
        z: centerZ + Math.sin(angle) * dist,
        colorBias: rand(), // Autumn or evergreen bias
      });
    }
    
    for (let i = 0; i < numTrees; i++) {
      // Pick a cluster to spawn near
      const cluster = clusterCenters[Math.floor(rand() * clusterCenters.length)];
      const clusterSpread = 6 + rand() * 14;
      
      const offsetAngle = rand() * Math.PI * 2;
      const offsetDist = rand() * clusterSpread;
      
      const x = cluster.x + Math.cos(offsetAngle) * offsetDist;
      const z = cluster.z + Math.sin(offsetAngle) * offsetDist;
      
      const distFromCenter = Math.sqrt((x - centerX) ** 2 + (z - centerZ) ** 2);
      const normalizedDist = distFromCenter / radius;
      
      // Trees on lower slopes (avoid peaks and water)
      if (normalizedDist > 0.1 && normalizedDist < 0.88) {
        // Use sampleHeight if available for accurate positioning
        let baseHeight: number;
        if (sampleHeight) {
          baseHeight = sampleHeight(x, z);
        } else {
          baseHeight = (1 - normalizedDist) * 8 + rand() * 2;
        }
        
        // Skip if would be in water
        if (baseHeight > 0.6) {
          const scale = 0.4 + rand() * 1.0;
          
          // Mix autumn and evergreen based on cluster bias
          let color: string;
          if (cluster.colorBias > 0.7 && rand() > 0.4) {
            color = autumnColors[Math.floor(rand() * autumnColors.length)];
          } else {
            color = treeColors[Math.floor(rand() * treeColors.length)];
          }
          
          treeList.push({
            id: `tree-${i}`,
            position: [x, baseHeight, z],
            scale,
            color,
            type: 'tree',
          });
        }
      }
    }
    
    // Add houses/cottages (8-14 scattered around)
    const numHouses = 8 + Math.floor(rand() * 6);
    for (let i = 0; i < numHouses; i++) {
      const angle = (i / numHouses) * Math.PI * 2 + rand() * 0.8;
      const dist = (0.3 + rand() * 0.35) * radius; // Mid-range distance
      
      const x = centerX + Math.cos(angle) * dist;
      const z = centerZ + Math.sin(angle) * dist;
      
      const distFromCenter = Math.sqrt((x - centerX) ** 2 + (z - centerZ) ** 2);
      const normalizedDist = distFromCenter / radius;
      
      if (normalizedDist > 0.2 && normalizedDist < 0.7) {
        let baseHeight: number;
        if (sampleHeight) {
          baseHeight = sampleHeight(x, z);
        } else {
          baseHeight = (1 - normalizedDist) * 8 + rand() * 2;
        }
        
        // Houses on flat-ish terrain, not in water
        if (baseHeight > 1.5 && baseHeight < 10) {
          treeList.push({
            id: `house-${i}`,
            position: [x, baseHeight, z],
            scale: 0.8 + rand() * 0.5,
            color: roofColors[Math.floor(rand() * roofColors.length)],
            type: 'house',
            rotation: rand() * Math.PI * 2,
          });
        }
      }
    }
    
    // Add bushes (80-110)
    const numBushes = 80 + Math.floor(rand() * 30);
    for (let i = 0; i < numBushes; i++) {
      const angle = rand() * Math.PI * 2;
      const dist = (0.12 + rand() * 0.72) * radius;
      
      const x = centerX + Math.cos(angle) * dist;
      const z = centerZ + Math.sin(angle) * dist;
      
      const distFromCenter = Math.sqrt((x - centerX) ** 2 + (z - centerZ) ** 2);
      const normalizedDist = distFromCenter / radius;
      
      if (normalizedDist > 0.08 && normalizedDist < 0.82) {
        let baseHeight: number;
        if (sampleHeight) {
          baseHeight = sampleHeight(x, z);
        } else {
          baseHeight = (1 - normalizedDist) * 8 + rand() * 2;
        }
        
        if (baseHeight > 0.4 && baseHeight < 14) {
          treeList.push({
            id: `bush-${i}`,
            position: [x, baseHeight, z],
            scale: 0.5 + rand() * 0.7,
            color: bushColors[Math.floor(rand() * bushColors.length)],
            type: 'bush',
          });
        }
      }
    }
    
    // Add rocks (40-65)
    const numRocks = 40 + Math.floor(rand() * 25);
    for (let i = 0; i < numRocks; i++) {
      const angle = rand() * Math.PI * 2;
      const dist = (0.05 + rand() * 0.88) * radius;
      
      const x = centerX + Math.cos(angle) * dist;
      const z = centerZ + Math.sin(angle) * dist;
      
      const distFromCenter = Math.sqrt((x - centerX) ** 2 + (z - centerZ) ** 2);
      const normalizedDist = distFromCenter / radius;
      
      let baseHeight: number;
      if (sampleHeight) {
        baseHeight = sampleHeight(x, z);
      } else {
        baseHeight = (1 - normalizedDist) * 8 + rand() * 2;
      }
      
      if (baseHeight > 0.2) {
        treeList.push({
          id: `rock-${i}`,
          position: [x, baseHeight, z],
          scale: 0.4 + rand() * 1.8,
          color: "#7f8c8d",
          type: 'rock',
        });
      }
    }
    
    return treeList;
  }, [bounds, seed, sampleHeight]);
  
  return (
    <group>
      {decorations.map((item) => {
        if (item.type === 'tree') {
          return (
            <LowPolyTree
              key={item.id}
              position={item.position}
              scale={item.scale}
              color={item.color}
            />
          );
        } else if (item.type === 'bush') {
          return (
            <Bush
              key={item.id}
              position={item.position}
              scale={item.scale}
              color={item.color}
            />
          );
        } else if (item.type === 'house') {
          return (
            <group key={item.id} rotation={[0, item.rotation ?? 0, 0]}>
              <Cottage
                position={item.position}
                scale={item.scale}
                roofColor={item.color}
              />
            </group>
          );
        } else {
          return (
            <Rock
              key={item.id}
              position={item.position}
              scale={item.scale}
            />
          );
        }
      })}
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
  filteredPoints,
  onSelect,
  onHover,
  focusPosition,
  onFocusComplete,
  showQueryPath,
  terrainData,
  highlightedIds,
  onRightClick,
}: {
  payload: TerrainPayload;
  filteredPoints: TerrainPoint[];
  onSelect: (point: TerrainPoint) => void;
  onHover?: (point: TerrainPoint | null, event?: React.MouseEvent) => void;
  focusPosition: [number, number, number] | null;
  onFocusComplete?: () => void;
  showQueryPath?: boolean;
  terrainData: ReturnType<typeof useTerrainData>;
  highlightedIds?: Set<string>;
  onRightClick?: (point: TerrainPoint, event: React.MouseEvent) => void;
}) {
  const bounds = useMemo(() => deriveBounds(payload), [payload]);
  const highlightPoints = useMemo(
    () => payload.highlights.map((highlight) => payload.points.find((p) => p.id === highlight.id)).filter(Boolean) as TerrainPoint[],
    [payload]
  );
  const meshSeed = payload.meta?.seed ?? 1337;
  
  // Use terrain data from props
  const { geometry, riverPoints, sampleHeight, terrainSize } = terrainData;

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
      
      {/* Sun in the sky */}
      <Sun position={[bounds.maxX + diag * 0.8, 55, bounds.minZ - diag * 0.3]} size={10} />
      
      {/* Ocean rim around the circular terrain */}
      <OceanRim bounds={bounds} terrainRadius={terrainSize / 2} />
      
      <SkyDecoration bounds={bounds} seed={meshSeed} />
      <TerrainSurface geometry={geometry} riverPoints={riverPoints} bounds={bounds} />
      <ForestDecoration bounds={bounds} seed={meshSeed} sampleHeight={sampleHeight} />
      {/* Query path showing the journey through ranked products */}
      <QueryPath points={filteredPoints} visible={showQueryPath ?? true} sampleHeight={sampleHeight} />
      <ProductMarkers points={filteredPoints} onSelect={onSelect} sampleHeight={sampleHeight} />
      <ProductBillboards points={filteredPoints} onSelect={onSelect} sampleHeight={sampleHeight} />
      <ProductLabels 
        points={filteredPoints} 
        onSelect={onSelect} 
        sampleHeight={sampleHeight} 
        onHover={onHover}
        highlightedIds={highlightedIds}
        onRightClick={onRightClick}
      />
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

  // Compute terrain bounds early so we can create height sampler at this level
  const bounds = useMemo(() => deriveBounds(payload), [payload]);
  const meshSeed = payload.meta?.seed ?? 1337;
  
  // Create terrain data including sampleHeight at this level so tour can use it
  const terrainData = useTerrainData(payload.points, bounds, meshSeed);
  const { sampleHeight } = terrainData;

  // Tour state
  const [tourActive, setTourActive] = useState(false);
  const [tourStep, setTourStep] = useState(0); // 0 = intro, 1-N = products, N+1 = outro
  const [focusPosition, setFocusPosition] = useState<[number, number, number] | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Filter state
  const [filters, setFilters] = useState<FilterState>(() => getDefaultFilters(payload.points));
  
  // Update filters when payload changes
  useEffect(() => {
    setFilters(getDefaultFilters(payload.points));
  }, [payload.points]);

  // Comparison state
  const [comparisonProducts, setComparisonProducts] = useState<TerrainPoint[]>([]);
  
  // Preview state
  const [previewProduct, setPreviewProduct] = useState<TerrainPoint | null>(null);
  const [previewPosition, setPreviewPosition] = useState<{ x: number; y: number } | undefined>();

  // Search state
  const [searchHighlightedIds, setSearchHighlightedIds] = useState<Set<string>>(new Set());
  
  // Score breakdown state
  const [scoreBreakdownProduct, setScoreBreakdownProduct] = useState<TerrainPoint | null>(null);
  const [scoreBreakdownPosition, setScoreBreakdownPosition] = useState<{ x: number; y: number } | undefined>();

  // Apply filters to get visible points
  const filteredPoints = useMemo(() => {
    return applyFilters(payload.points, filters);
  }, [payload.points, filters]);

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

  // Focus camera on current tour product - using terrain-sampled height
  useEffect(() => {
    if (currentTourProduct) {
      const terrainHeight = sampleHeight(currentTourProduct.position[0], currentTourProduct.position[2]);
      const y = Math.max(terrainHeight, currentTourProduct.height);
      setFocusPosition([
        currentTourProduct.position[0],
        y,
        currentTourProduct.position[2],
      ]);
    }
  }, [currentTourProduct, sampleHeight]);

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
    // Focus camera on selected point using terrain-sampled height
    const terrainHeight = sampleHeight(point.position[0], point.position[2]);
    const y = Math.max(terrainHeight, point.height);
    setFocusPosition([point.position[0], y, point.position[2]]);
    
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

  // Handle hover for product preview
  const handleProductHover = useCallback((point: TerrainPoint | null, event?: React.MouseEvent) => {
    if (point && event) {
      setPreviewProduct(point);
      setPreviewPosition({ x: event.clientX, y: event.clientY });
    } else {
      setPreviewProduct(null);
      setPreviewPosition(undefined);
    }
  }, []);

  // Handle right-click for score breakdown
  const handleProductRightClick = useCallback((point: TerrainPoint, event: React.MouseEvent) => {
    setScoreBreakdownProduct(point);
    setScoreBreakdownPosition({ x: event.clientX, y: event.clientY });
  }, []);

  // Handle search results
  const handleSearchResult = useCallback((ids: Set<string>) => {
    setSearchHighlightedIds(ids);
  }, []);

  // Handle search focus (camera moves to product)
  const handleSearchFocus = useCallback((point: TerrainPoint) => {
    const terrainHeight = sampleHeight(point.position[0], point.position[2]);
    const y = Math.max(terrainHeight, point.height);
    setFocusPosition([point.position[0], y, point.position[2]]);
  }, [sampleHeight]);

  // Comparison handlers
  const handleAddToComparison = useCallback((point: TerrainPoint) => {
    setComparisonProducts((prev) => {
      // Check if already in comparison
      if (prev.some((p) => p.id === point.id)) {
        return prev;
      }
      // Max 3 products
      if (prev.length >= 3) {
        return prev;
      }
      return [...prev, point];
    });
  }, []);

  const handleRemoveFromComparison = useCallback((id: string) => {
    setComparisonProducts((prev) => prev.filter((p) => p.id !== id));
  }, []);

  const handleClearComparison = useCallback(() => {
    setComparisonProducts([]);
  }, []);

  // Combined select handler - adds to comparison and shows preview
  const handleProductSelect = useCallback((point: TerrainPoint) => {
    handleSelect(point);
    handleAddToComparison(point);
  }, [handleAddToComparison]);

  const handleFocusComplete = useCallback(() => {
    // Optional: auto-clear focus after animation
  }, []);

  useEffect(() => {
    Streamlit.setFrameHeight();
  }, [payload, props.width]);

  // Fullscreen toggle handler
  const toggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;
    
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().then(() => {
        setIsFullscreen(true);
      }).catch((err) => {
        console.error('Fullscreen error:', err);
      });
    } else {
      document.exitFullscreen().then(() => {
        setIsFullscreen(false);
      }).catch((err) => {
        console.error('Exit fullscreen error:', err);
      });
    }
  }, []);

  // Listen for fullscreen changes (e.g., user presses Escape)
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  return (
    <div className="terrain-shell" ref={containerRef}>
      <Scene 
        payload={payload}
        filteredPoints={filteredPoints}
        onSelect={handleProductSelect}
        onHover={handleProductHover}
        focusPosition={focusPosition}
        onFocusComplete={handleFocusComplete}
        showQueryPath={true}
        terrainData={terrainData}
        highlightedIds={searchHighlightedIds}
        onRightClick={handleProductRightClick}
      />
      <NarrationPanel payload={payload} />
      <ControlIndicators visible={!tourActive} />
      
      {/* Search Panel */}
      <SearchPanel
        points={filteredPoints}
        onSearchResult={handleSearchResult}
        onFocusProduct={handleSearchFocus}
        visible={!tourActive}
      />
      
      {/* Filter Panel */}
      <FilterPanel
        points={payload.points}
        filters={filters}
        onFiltersChange={setFilters}
        visible={!tourActive}
      />
      
      {/* Product Preview on hover */}
      <ProductPreview
        product={previewProduct}
        position={previewPosition}
        onClose={() => setPreviewProduct(null)}
      />
      
      {/* Score Breakdown on right-click */}
      <ScoreBreakdown
        product={scoreBreakdownProduct}
        position={scoreBreakdownPosition}
        onClose={() => {
          setScoreBreakdownProduct(null);
          setScoreBreakdownPosition(undefined);
        }}
      />
      
      {/* Comparison Panel */}
      <ComparisonPanel
        selectedProducts={comparisonProducts}
        onRemove={handleRemoveFromComparison}
        onClear={handleClearComparison}
        visible={!tourActive && comparisonProducts.length > 0}
      />
      
      <button
        className="fullscreen-btn"
        onClick={toggleFullscreen}
        title={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
      >
        {isFullscreen ? (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3" />
          </svg>
        ) : (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
          </svg>
        )}
      </button>
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
