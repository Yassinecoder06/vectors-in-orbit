import { useMemo } from "react";
import * as THREE from "three";
import { Line } from "@react-three/drei";
import type { TerrainPoint } from "./types";

// Height sampler function type
type HeightSampler = (x: number, z: number) => number;

interface QueryPathProps {
  points: TerrainPoint[];
  visible?: boolean;
  sampleHeight?: HeightSampler;
}

/**
 * QueryPath - Line connecting products in rank order
 * Shows the "journey" through search results from best (#1) to worst
 * 
 * Features:
 * - Gradient color from gold (#1) to gray (last)
 * - Lifted slightly above terrain to avoid z-fighting
 * - Smooth curve through all product positions
 */
const QueryPath = ({ points, visible = true, sampleHeight }: QueryPathProps) => {
  // Sort points by rank (they should already be sorted, but ensure it)
  const sortedPoints = useMemo(() => {
    return [...points].sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0));
  }, [points]);

  // Create path points and colors
  const { pathPoints, pathColors } = useMemo(() => {
    if (sortedPoints.length < 2) {
      return { pathPoints: [], pathColors: [] };
    }

    const pts: THREE.Vector3[] = [];
    const cols: THREE.Color[] = [];
    const totalPoints = sortedPoints.length;

    // Gold color for #1, fading to silver/gray for last
    const startColor = new THREE.Color("#ffd700"); // Gold
    const endColor = new THREE.Color("#708090");   // Slate gray

    sortedPoints.forEach((point, index) => {
      // Sample terrain height if sampler provided, otherwise use point height
      const terrainHeight = sampleHeight 
        ? sampleHeight(point.position[0], point.position[2])
        : point.position[1];
      const baseY = Math.max(terrainHeight, point.height);
      
      // Lift path slightly above the product markers
      const lift = 1.8;
      pts.push(new THREE.Vector3(
        point.position[0],
        baseY + lift,
        point.position[2]
      ));

      // Gradient color based on rank
      const t = index / Math.max(totalPoints - 1, 1);
      const color = new THREE.Color().lerpColors(startColor, endColor, t);
      cols.push(color);
    });

    return { pathPoints: pts, pathColors: cols };
  }, [sortedPoints, sampleHeight]);

  if (!visible || pathPoints.length < 2) {
    return null;
  }

  return (
    <Line
      points={pathPoints}
      vertexColors={pathColors}
      lineWidth={1.5}
      transparent
      opacity={0.35}
      dashed
      dashScale={2}
      dashSize={1}
      gapSize={0.5}
      depthTest={false}
      renderOrder={999}
    />
  );
};

export default QueryPath;
