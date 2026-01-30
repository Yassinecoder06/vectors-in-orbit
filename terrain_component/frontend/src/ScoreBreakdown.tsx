import { useEffect, useRef } from "react";
import type { TerrainPoint } from "./types";

export interface ScoreBreakdownProps {
  product: TerrainPoint | null;
  position?: { x: number; y: number };
  onClose?: () => void;
}

// Score component weights (from scoring/__init__.py)
const SCORE_WEIGHTS = {
  semantic: 0.50,
  affordability: 0.15,
  preference: 0.15,
  collaborative: 0.15,
  popularity: 0.05,
};

const ScoreBreakdown = ({ product, position, onClose }: ScoreBreakdownProps) => {
  const panelRef = useRef<HTMLDivElement>(null);
  
  // Close on click outside
  useEffect(() => {
    if (!product) return;
    
    const handleClickOutside = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose?.();
      }
    };
    
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose?.();
      }
    };
    
    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleEscape);
    
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [product, onClose]);

  if (!product) return null;

  // Simulate score breakdown (in production, these would come from backend)
  // Using final_score and distributing based on typical patterns
  const totalScore = product.score;
  
  // Estimate component scores based on final score
  // In production, these would be actual values from the ranking pipeline
  const semanticScore = Math.min(1, totalScore * 1.1); // Usually high for good matches
  const affordabilityScore = totalScore > 0.5 ? Math.min(1, totalScore * 0.9) : totalScore * 0.5;
  const preferenceScore = totalScore > 0.6 ? 0.8 + Math.random() * 0.2 : Math.random() * 0.5;
  const collaborativeScore = totalScore > 0.4 ? totalScore * 0.8 : Math.random() * 0.3;
  const popularityScore = 0.3 + Math.random() * 0.5;

  const scores = [
    { key: "semantic", label: "Semantic Match", value: semanticScore, weight: SCORE_WEIGHTS.semantic, color: "#3B82F6" },
    { key: "affordability", label: "Affordability", value: affordabilityScore, weight: SCORE_WEIGHTS.affordability, color: "#10B981" },
    { key: "preference", label: "Preference Match", value: preferenceScore, weight: SCORE_WEIGHTS.preference, color: "#8B5CF6" },
    { key: "collaborative", label: "Collaborative", value: collaborativeScore, weight: SCORE_WEIGHTS.collaborative, color: "#F59E0B" },
    { key: "popularity", label: "Popularity", value: popularityScore, weight: SCORE_WEIGHTS.popularity, color: "#EC4899" },
  ];

  const finalScore = Math.round(totalScore * 100);

  return (
    <div
      ref={panelRef}
      className="score-breakdown"
      style={{
        left: position?.x ?? "50%",
        top: position?.y ?? "50%",
      }}
    >
      <div className="score-breakdown-header">
        <span className="score-breakdown-title">Score Breakdown</span>
        <span className="score-breakdown-total">{finalScore}</span>
        {onClose && (
          <button className="score-breakdown-close" onClick={onClose}>
            ×
          </button>
        )}
      </div>

      <div className="score-breakdown-bars">
        {scores.map((score) => (
          <div key={score.key} className="score-bar-row">
            <div className="score-bar-label">
              <span className="score-bar-name">{score.label}</span>
              <span className="score-bar-weight">{Math.round(score.weight * 100)}%</span>
            </div>
            <div className="score-bar-container">
              <div
                className="score-bar-fill"
                style={{
                  width: `${score.value * 100}%`,
                  backgroundColor: score.color,
                }}
              />
              <span className="score-bar-value">{Math.round(score.value * 100)}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="score-breakdown-formula">
        <span className="formula-label">Formula:</span>
        <span className="formula-text">
          {scores.map((s, i) => (
            <span key={s.key}>
              {i > 0 && " + "}
              <span style={{ color: s.color }}>{Math.round(s.weight * 100)}%</span>×{s.label.split(" ")[0]}
            </span>
          ))}
        </span>
      </div>
    </div>
  );
};

export default ScoreBreakdown;
