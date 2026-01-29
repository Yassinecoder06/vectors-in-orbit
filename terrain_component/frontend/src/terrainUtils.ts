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

// Multi-frequency noise for natural terrain variation
function terrainNoise(x: number, z: number, seed: number): number {
  const f1 = 0.03;
  const f2 = 0.07;
  const f3 = 0.015;
  
  const noise1 = Math.sin(x * f1 + seed * 0.01) * Math.cos(z * f1 - seed * 0.008) * 0.4;
  const noise2 = Math.sin(x * f2 - 1.23) * Math.cos(z * f2 + 0.87) * 0.25;
  const noise3 = Math.sin((x + z) * f3 + seed * 0.005) * 0.15;
  const microNoise = Math.sin(x * 0.15 + 3.7) * Math.cos(z * 0.18 - 2.1) * 0.08;
  
  return noise1 + noise2 + noise3 + microNoise;
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
 * Products at higher ranks (better matches) become taller peaks
 */
export function generateMountainsFromProducts(
  points: TerrainPoint[],
  bounds: TerrainBounds
): { mountains: Mountain[]; hills: Mountain[] } {
  const mountains: Mountain[] = [];
  const hills: Mountain[] = [];
  
  if (points.length === 0) {
    // Default central mountain
    mountains.push({
      x: 0,
      z: 0,
      height: 20,
      radius: 25,
      slopeFactor: 0.6,
    });
    return { mountains, hills };
  }
  
  // Sort by rank (best first)
  const sortedPoints = [...points].sort((a, b) => (a.rank ?? 999) - (b.rank ?? 999));
  
  // Top products become main mountain peaks
  const topProducts = sortedPoints.slice(0, Math.min(8, sortedPoints.length));
  const restProducts = sortedPoints.slice(Math.min(8, sortedPoints.length));
  
  topProducts.forEach((point, idx) => {
    const rankFactor = 1 - idx / Math.max(topProducts.length - 1, 1);
    mountains.push({
      x: point.position[0],
      z: point.position[2],
      height: 8 + rankFactor * 18, // Height 8-26 based on rank
      radius: 12 + rankFactor * 8, // Radius 12-20
      slopeFactor: 0.5 + rankFactor * 0.3, // Sharper peaks for top items
    });
  });
  
  // Rest become smaller hills
  restProducts.forEach((point, idx) => {
    hills.push({
      x: point.position[0],
      z: point.position[2],
      height: 3 + point.height * 0.3,
      radius: 6 + Math.random() * 4,
      slopeFactor: 0.7,
    });
  });
  
  return { mountains, hills };
}

/**
 * Generate river path through low-elevation areas
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
  
  // Create winding river from one corner through the center to opposite corner
  const points: RiverPoint[] = [];
  const segments = 12;
  
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    
    // Start NW, wind through center, end SE
    const baseX = centerX + (t - 0.5) * width * 0.9;
    const baseZ = centerZ + (0.5 - t) * depth * 0.9;
    
    // Add sinusoidal winding
    const winding = Math.sin(t * Math.PI * 2.5) * (width * 0.15);
    
    // Find lowest nearby point (avoid mountain peaks)
    let bestX = baseX + winding;
    let bestZ = baseZ;
    let lowestInfluence = Infinity;
    
    for (let dx = -10; dx <= 10; dx += 5) {
      for (let dz = -10; dz <= 10; dz += 5) {
        const testX = baseX + dx;
        const testZ = baseZ + dz;
        
        // Calculate mountain influence at this point
        let influence = 0;
        mountains.forEach(m => {
          const dist = Math.sqrt((testX - m.x) ** 2 + (testZ - m.z) ** 2);
          if (dist < m.radius * 1.5) {
            influence += m.height * Math.exp(-dist / m.radius);
          }
        });
        
        if (influence < lowestInfluence) {
          lowestInfluence = influence;
          bestX = testX;
          bestZ = testZ;
        }
      }
    }
    
    points.push({ x: bestX, z: bestZ });
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
      
      // Carve river through terrain
      if (riverPoints && riverPoints.length > 1) {
        for (let k = 0; k < riverPoints.length - 1; k++) {
          const p1 = riverPoints[k];
          const p2 = riverPoints[k + 1];
          const dist = distanceToLineSegment(x, z, p1.x, p1.z, p2.x, p2.z);
          
          const riverWidth = 3.5;
          const bankWidth = riverWidth * 2;
          
          if (dist < bankWidth) {
            if (dist < riverWidth * 0.6) {
              // River center - below water level
              elevation = Math.min(elevation, -1.2);
            } else if (dist < riverWidth) {
              // River bed transition
              const t = (dist - riverWidth * 0.6) / (riverWidth * 0.4);
              elevation = Math.min(elevation, -1.2 + t * 1.5);
            } else {
              // River banks - gentle slope down
              const t = (dist - riverWidth) / (bankWidth - riverWidth);
              const bankFactor = 1 - (1 - t) * 0.4;
              elevation *= bankFactor;
            }
          }
        }
      }
      
      // Ensure edges go to water level for circular terrain
      if (circularBoundary && distFromCenter > radius * 0.95) {
        elevation = Math.min(elevation, -1.5);
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
