export type TerrainPoint = {
  id: string;
  position: [number, number, number];
  height: number;
  color: string;
  price: number;
  price_normalized?: number;
  brand: string;
  category: string;
  name: string;
  description?: string;
  score: number;
  risk_tolerance: number;
  imageUrl?: string;
  rank?: number;  // 1-indexed rank (1 = best match)
};

export type TerrainBounds = {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
};

export type TerrainMeta = {
  bounds?: TerrainBounds;
  price_range?: { min: number; max: number };
  height_scale?: number;
  sample_size?: number;
  seed?: number;
  generated_at?: number;
  mode?: string;
};

export type Highlight = {
  id: string;
  label: string;
  position: [number, number, number];
  price: number;
  brand: string;
  category: string;
  score: number;
};

export type TerrainPayload = {
  points: TerrainPoint[];
  highlights: Highlight[];
  meta?: TerrainMeta;
};
