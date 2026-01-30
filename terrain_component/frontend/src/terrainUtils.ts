/**
 * Terrain generation utilities inspired by Vector Vintage
 * Creates procedural alpine landscape with multiple mountain peaks, valleys, and rivers
 */

import * as THREE from "three";
import type { TerrainPoint, TerrainBounds } from "./types";

// Grid configuration
const GRID_SIZE = 180;
const TERRAIN_WIDTH = 150;
const TERRAIN_HEIGHT = 150;

// Gaussian function for natural mountain peaks
function gaussian(
  x: number,
  z: number,
  cx: number,
  cz: number,
  height: number,
  radius: number,
  slopeFactor: number = 0.7
): number {
  const dx = x - cx;
  const dz = z - cz;
  const dist2 = dx * dx + dz * dz;
  const sigma = radius * slopeFactor;
  return height * Math.exp(-dist2 / (2 * sigma * sigma));
}

// Multi-frequency noise for natural terrain variation - enhanced for rougher texture
function terrainNoise(x: number, z: number, seed: number): number {
  // Low frequency - large terrain features
  const f1 = 0.025;
  const noise1 = Math.sin(x * f1 + seed * 0.01) * Math.cos(z * f1 - seed * 0.008) * 0.5;
  
  // Medium frequency - hills and bumps
  const f2 = 0.06;
  const noise2 = Math.sin(x * f2 - 1.23) * Math.cos(z * f2 + 0.87) * 0.35;
  
  // Higher frequency - terrain texture
  const f3 = 0.12;
  const noise3 = Math.sin(x * f3 + 2.1) * Math.cos(z * f3 - 1.4) * 0.25;
  
  // Fine detail - roughness
  const f4 = 0.2;
  const noise4 = Math.sin(x * f4 - 0.7) * Math.cos(z * f4 + 3.2) * 0.18;
  
  // Micro texture - graininess
  const f5 = 0.35;
  const microNoise = Math.sin(x * f5 + 3.7) * Math.cos(z * f5 - 2.1) * 0.12;
  
  // Ultra-fine noise for rough look
  const f6 = 0.5;
  const ultraFine = Math.sin(x * f6 + 1.3) * Math.cos(z * f6 + 0.9) * 0.08;
  
  return noise1 + noise2 + noise3 + noise4 + microNoise + ultraFine;
}

// Distance from point to line segment (for river carving)
function distanceToLineSegment(
  px: number, py: number,
  x1: number, y1: number,
  x2: number, y2: number
): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len2 = dx * dx + dy * dy;
  
  if (len2 < 1e-6) {
    return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
  }
  
  let t = ((px - x1) * dx + (py - y1) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  
  const nearX = x1 + t * dx;
  const nearY = y1 + t * dy;
  
  return Math.sqrt((px - nearX) ** 2 + (py - nearY) ** 2);
}

export interface Mountain {
  x: number;
  z: number;
  height: number;
  radius: number;
  slopeFactor?: number;
}

export interface RiverPoint {
  x: number;
  z: number;
}

export interface TerrainConfig {
  width: number;
  height: number;
  resolution: number;
  mountains: Mountain[];
  hills: Mountain[];
  riverPoints?: RiverPoint[];
  seed: number;
  circularBoundary?: boolean;
  boundaryRadius?: number;
}

/**
 * Generate mountains from product positions
 * Products with higher scores become taller peaks (score-based height)
 * Best product is placed at the central peak
 */
export function generateMountainsFromProducts(
  points: TerrainPoint[],
  bounds: TerrainBounds
): { mountains: Mountain[]; hills: Mountain[] } {
  const mountains: Mountain[] = [];
  const hills: Mountain[] = [];
  
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerZ = (bounds.minZ + bounds.maxZ) / 2;
  
  // Find the best product's height (score-based) for the central peak
  const bestProductHeight = points.length > 0 
    ? Math.max(...points.map(p => p.height)) * 0.6  // Reduce height by 40%
    : 12;
  
  // Main central mountain - very wide and gentle slopes
  mountains.push({
    x: centerX,
    z: centerZ,
    height: bestProductHeight, // Reduced height
    radius: 70,  // Much wider base
    slopeFactor: 0.9,  // Very gentle, rounded slopes
  });
  
  if (points.length === 0) {
    return { mountains, hills };
  }
  
  // Sort by score (highest first) - score determines height
  const sortedPoints = [...points].sort((a, b) => b.score - a.score);
  
  // Best product gets a peak at the very center (on top of the mountain)
  // Other top products become secondary peaks around the main one
  const topProducts = sortedPoints.slice(0, Math.min(6, sortedPoints.length));
  const restProducts = sortedPoints.slice(Math.min(6, sortedPoints.length));
  
  topProducts.forEach((point, idx) => {
    if (idx === 0) {
      // Best product - gentle dome at center
      mountains.push({
        x: centerX,
        z: centerZ,
        height: point.height * 0.5 + 1, // Much shorter peak
        radius: 15,  // Wide peak base
        slopeFactor: 0.85, // Very gentle peak
      });
    } else {
      // Other top products - gentle mounds
      const scoreFactor = point.score; // 0.0 to 1.0
      mountains.push({
        x: point.position[0],
        z: point.position[2],
        height: point.height * 0.5 + 0.5, // Much shorter
        radius: 15 + scoreFactor * 10, // Wide bases
        slopeFactor: 0.75 + scoreFactor * 0.15, // Gentle slopes
      });
    }
  });
  
  // Rest become smaller hills - height based on their scores
  restProducts.forEach((point, idx) => {
    hills.push({
      x: point.position[0],
      z: point.position[2],
      height: point.height * 0.8, // Slightly below product height (score-based)
      radius: 3 + point.score * 3, // Score affects hill size
      slopeFactor: 0.6,
    });
  });
  
  return { mountains, hills };
}

/**
 * Generate river path around the mountain perimeter (not through it)
 */
export function generateRiverPath(
  mountains: Mountain[],
  bounds: TerrainBounds
): RiverPoint[] {
  if (mountains.length === 0) {
    return [];
  }
  
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerZ = (bounds.minZ + bounds.maxZ) / 2;
  const width = bounds.maxX - bounds.minX;
  const depth = bounds.maxZ - bounds.minZ;
  const radius = Math.min(width, depth) / 2;
  
  // Create river that flows around the mountain at the outer edge
  const points: RiverPoint[] = [];
  const segments = 24; // More segments for smoother curve
  const riverRadius = radius * 0.85; // River flows at 85% of terrain radius
  
  // River flows in an arc around the bottom/sides of the mountain (not a full circle)
  const startAngle = Math.PI * 0.6;  // Start from lower-left
  const endAngle = Math.PI * 2.4;    // End at lower-right (wrapping around bottom)
  
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const angle = startAngle + t * (endAngle - startAngle);
    
    // Add some natural winding variation
    const radiusVariation = riverRadius + Math.sin(t * Math.PI * 4) * (radius * 0.05);
    
    const x = centerX + Math.cos(angle) * radiusVariation;
    const z = centerZ + Math.sin(angle) * radiusVariation;
    
    points.push({ x, z });
  }
  
  return points;
}

/**
 * Build terrain height grid with mountains, hills, and river
 */
export function buildHeightGrid(config: TerrainConfig): Float32Array {
  const { width, height, resolution, mountains, hills, riverPoints, seed, circularBoundary, boundaryRadius } = config;
  
  const grid = new Float32Array(resolution * resolution);
  const stepX = width / resolution;
  const stepZ = height / resolution;
  const startX = -width / 2;
  const startZ = -height / 2;
  const radius = boundaryRadius ?? Math.min(width, height) / 2;
  
  for (let iz = 0; iz < resolution; iz++) {
    for (let ix = 0; ix < resolution; ix++) {
      const idx = iz * resolution + ix;
      const x = startX + ix * stepX;
      const z = startZ + iz * stepZ;
      
      const distFromCenter = Math.sqrt(x * x + z * z);
      
      // Start with base terrain noise
      let elevation = terrainNoise(x, z, seed) * 2;
      
      // Apply circular boundary falloff
      if (circularBoundary && distFromCenter > radius * 0.7) {
        const normalizedDist = (distFromCenter - radius * 0.7) / (radius * 0.2);
        const falloff = Math.exp(-(normalizedDist * normalizedDist));
        elevation *= falloff;
      }
      
      // Add mountain peaks
      mountains.forEach((m, midx) => {
        const slopeFactor = m.slopeFactor ?? (0.5 + Math.sin(midx * 2.1) * 0.3);
        const mountainHeight = gaussian(x, z, m.x, m.z, m.height, m.radius, slopeFactor);
        
        // Apply boundary falloff to mountains too
        let factor = 1;
        if (circularBoundary && distFromCenter > radius * 0.6) {
          const normDist = (distFromCenter - radius * 0.6) / (radius * 0.25);
          factor = Math.exp(-(normDist * normDist));
        }
        
        elevation += mountainHeight * factor;
        
        // Add foothills between nearby mountains
        mountains.forEach((nearM, nearIdx) => {
          if (nearIdx !== midx) {
            const dist = Math.sqrt((m.x - nearM.x) ** 2 + (m.z - nearM.z) ** 2);
            if (dist < 30 && dist > 0) {
              const influence = (30 - dist) / 30;
              const bridgeHeight = gaussian(
                x, z,
                (m.x + nearM.x) / 2,
                (m.z + nearM.z) / 2,
                Math.min(m.height, nearM.height) * 0.25 * influence,
                dist / 2 + 5,
                1.0
              );
              elevation += bridgeHeight * factor * 0.4;
            }
          }
        });
      });
      
      // Add smaller hills
      hills.forEach((h, hidx) => {
        const slopeFactor = 0.5 + Math.sin(hidx * 1.8) * 0.2;
        const hillHeight = gaussian(x, z, h.x, h.z, h.height, h.radius, slopeFactor);
        
        let factor = 1;
        if (circularBoundary && distFromCenter > radius * 0.6) {
          const normDist = (distFromCenter - radius * 0.6) / (radius * 0.3);
          factor = Math.exp(-(normDist * normDist));
        }
        
        elevation += hillHeight * factor;
      });
      
      // Carve river around the terrain perimeter (not through the mountain)
      if (riverPoints && riverPoints.length > 1) {
        for (let k = 0; k < riverPoints.length - 1; k++) {
          const p1 = riverPoints[k];
          const p2 = riverPoints[k + 1];
          const dist = distanceToLineSegment(x, z, p1.x, p1.z, p2.x, p2.z);
          
          const riverWidth = 4;
          const bankWidth = riverWidth * 2.5;
          
          // Only carve river in the outer area (not near center mountain)
          const distFromTerrainCenter = Math.sqrt(x * x + z * z);
          const isOuterArea = distFromTerrainCenter > radius * 0.6;
          
          if (dist < bankWidth && isOuterArea) {
            if (dist < riverWidth * 0.5) {
              // River center - below water level
              elevation = Math.min(elevation, -1.0);
            } else if (dist < riverWidth) {
              // River bed transition
              const t = (dist - riverWidth * 0.5) / (riverWidth * 0.5);
              elevation = Math.min(elevation, -1.0 + t * 1.2);
            } else {
              // River banks - gentle slope down
              const t = (dist - riverWidth) / (bankWidth - riverWidth);
              const bankFactor = 1 - (1 - t) * 0.3;
              elevation *= bankFactor;
            }
          }
        }
      }
      
      // Ensure edges go to water level for circular terrain (ocean around the island)
      if (circularBoundary && distFromCenter > radius * 0.92) {
        const edgeFactor = (distFromCenter - radius * 0.92) / (radius * 0.08);
        elevation = elevation * (1 - edgeFactor) + (-1.5) * edgeFactor;
      }
      
      grid[idx] = elevation;
    }
  }
  
  return grid;
}

/**
 * Build THREE.js geometry from height grid with vertex colors
 */
export function buildTerrainMesh(
  grid: Float32Array,
  config: TerrainConfig
): THREE.BufferGeometry {
  const { width, height, resolution, circularBoundary, boundaryRadius } = config;
  
  const geometry = new THREE.BufferGeometry();
  const vertices: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const normals: number[] = [];
  
  const stepX = width / (resolution - 1);
  const stepZ = height / (resolution - 1);
  const startX = -width / 2;
  const startZ = -height / 2;
  const radius = boundaryRadius ?? Math.min(width, height) / 2;
  
  // Find height range for color mapping
  let minH = Infinity, maxH = -Infinity;
  grid.forEach(h => {
    if (h > -10) { // Ignore deep water
      minH = Math.min(minH, h);
      maxH = Math.max(maxH, h);
    }
  });
  const rangeH = Math.max(maxH - minH, 1);
  
  // Alpine color palette
  const colorWater = new THREE.Color('#5DADE2');   // River/lake blue
  const colorGrass = new THREE.Color('#90EE90');   // Light grass green
  const colorMeadow = new THREE.Color('#7CB342');  // Alpine meadow
  const colorRock = new THREE.Color('#9E9E9E');    // Gray rock
  const colorSnow = new THREE.Color('#FFFFFF');    // Snow white
  
  const color = new THREE.Color();
  
  // Build vertices and colors
  for (let iz = 0; iz < resolution; iz++) {
    for (let ix = 0; ix < resolution; ix++) {
      const idx = iz * resolution + ix;
      const x = startX + ix * stepX;
      const z = startZ + iz * stepZ;
      const y = grid[idx];
      
      const distFromCenter = Math.sqrt(x * x + z * z);
      
      // Skip vertices outside circular boundary
      if (circularBoundary && distFromCenter > radius * 1.05) {
        vertices.push(x, -2, z); // Place at water level
        colors.push(colorWater.r, colorWater.g, colorWater.b);
        continue;
      }
      
      vertices.push(x, y, z);
      
      // Determine color based on height
      if (y < -0.5) {
        // Water
        color.copy(colorWater);
      } else {
        const normalizedH = Math.max(0, (y - minH) / rangeH);
        
        if (normalizedH < 0.3) {
          // Grass to meadow
          color.lerpColors(colorGrass, colorMeadow, normalizedH / 0.3);
        } else if (normalizedH < 0.6) {
          // Meadow to rock
          color.lerpColors(colorMeadow, colorRock, (normalizedH - 0.3) / 0.3);
        } else if (normalizedH < 0.8) {
          // Rock
          color.copy(colorRock);
        } else {
          // Rock to snow
          color.lerpColors(colorRock, colorSnow, (normalizedH - 0.8) / 0.2);
        }
      }
      
      colors.push(color.r, color.g, color.b);
    }
  }
  
  // Build indices for triangles
  for (let iz = 0; iz < resolution - 1; iz++) {
    for (let ix = 0; ix < resolution - 1; ix++) {
      const a = iz * resolution + ix;
      const b = a + 1;
      const c = a + resolution;
      const d = c + 1;
      
      indices.push(a, c, b);
      indices.push(b, c, d);
    }
  }
  
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  
  return geometry;
}

/**
 * Sample terrain height at a specific world position
 */
export function sampleTerrainHeight(
  grid: Float32Array,
  worldX: number,
  worldZ: number,
  width: number,
  height: number,
  resolution: number
): number {
  const normX = (worldX / width + 0.5) * (resolution - 1);
  const normZ = (worldZ / height + 0.5) * (resolution - 1);
  
  const ix = Math.floor(normX);
  const iz = Math.floor(normZ);
  
  if (ix < 0 || ix >= resolution - 1 || iz < 0 || iz >= resolution - 1) {
    return 0;
  }
  
  const fx = normX - ix;
  const fz = normZ - iz;
  
  const h00 = grid[iz * resolution + ix];
  const h10 = grid[iz * resolution + ix + 1];
  const h01 = grid[(iz + 1) * resolution + ix];
  const h11 = grid[(iz + 1) * resolution + ix + 1];
  
  // Bilinear interpolation
  const h0 = h00 * (1 - fx) + h10 * fx;
  const h1 = h01 * (1 - fx) + h11 * fx;
  
  return h0 * (1 - fz) + h1 * fz;
}
