import { useMemo } from "react";
import * as THREE from "three";
import { Line } from "@react-three/drei";
import type { TerrainPoint } from "./types";

interface QueryPathProps {
  points: TerrainPoint[];
  visible?: boolean;
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
const QueryPath = ({ points, visible = true }: QueryPathProps) => {
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
      // Lift path slightly above the product markers
      const lift = 1.8;
      pts.push(new THREE.Vector3(
        point.position[0],
        point.position[1] + lift,
        point.position[2]
      ));

      // Gradient color based on rank
      const t = index / Math.max(totalPoints - 1, 1);
      const color = new THREE.Color().lerpColors(startColor, endColor, t);
      cols.push(color);
    });

    return { pathPoints: pts, pathColors: cols };
  }, [sortedPoints]);

  if (!visible || pathPoints.length < 2) {
    return null;
  }

  return (
    <Line
      points={pathPoints}
      vertexColors={pathColors}
      lineWidth={3}
      transparent
      opacity={0.75}
      dashed
      dashScale={2}
      dashSize={1}
      gapSize={0.5}
    />
  );
};

export default QueryPath;
