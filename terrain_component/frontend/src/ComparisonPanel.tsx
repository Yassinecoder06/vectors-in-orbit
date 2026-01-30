import { useMemo } from "react";
import type { TerrainPoint } from "./types";

interface ComparisonPanelProps {
  selectedProducts: TerrainPoint[];
  onRemove: (id: string) => void;
  onClear: () => void;
  visible?: boolean;
}

const ComparisonPanel = ({ selectedProducts, onRemove, onClear, visible = true }: ComparisonPanelProps) => {
  if (!visible || selectedProducts.length === 0) return null;

  const maxProducts = 3;

  // Get comparison metrics
  const metrics = useMemo(() => {
    if (selectedProducts.length === 0) return null;
    
    const prices = selectedProducts.map((p) => p.price);
    const scores = selectedProducts.map((p) => p.score * 100);
    
    return {
      avgPrice: prices.reduce((a, b) => a + b, 0) / prices.length,
      minPrice: Math.min(...prices),
      maxPrice: Math.max(...prices),
      avgScore: scores.reduce((a, b) => a + b, 0) / scores.length,
      bestScore: Math.max(...scores),
    };
  }, [selectedProducts]);

  return (
    <div className="comparison-panel">
      <div className="comparison-header">
        <h3>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="20" x2="18" y2="10" />
            <line x1="12" y1="20" x2="12" y2="4" />
            <line x1="6" y1="20" x2="6" y2="14" />
          </svg>
          Compare ({selectedProducts.length}/{maxProducts})
        </h3>
        {selectedProducts.length > 0 && (
          <button className="comparison-clear" onClick={onClear}>
            Clear All
          </button>
        )}
      </div>

      <div className="comparison-products">
        {selectedProducts.map((product) => (
          <div key={product.id} className="comparison-product">
            <div className="comparison-product-header">
              {product.imageUrl && (
                <img
                  src={product.imageUrl}
                  alt={product.name}
                  className="comparison-product-image"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
              )}
              <button
                className="comparison-remove"
                onClick={() => onRemove(product.id)}
                title="Remove from comparison"
              >
                ×
              </button>
            </div>
            <div className="comparison-product-info">
              <span className="comparison-product-name">{product.name}</span>
              <div className="comparison-product-meta">
                <span className="comparison-price">${product.price.toFixed(2)}</span>
                <span className="comparison-score">{Math.round(product.score * 100)}</span>
              </div>
              <span className="comparison-brand">{product.brand}</span>
              <span className="comparison-category">{product.category}</span>
            </div>
          </div>
        ))}

        {/* Empty slots */}
        {Array.from({ length: maxProducts - selectedProducts.length }).map((_, i) => (
          <div key={`empty-${i}`} className="comparison-product empty">
            <div className="comparison-empty-text">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="16" />
                <line x1="8" y1="12" x2="16" y2="12" />
              </svg>
              <span>Click a product to add</span>
            </div>
          </div>
        ))}
      </div>

      {selectedProducts.length >= 2 && metrics && (
        <div className="comparison-summary">
          <div className="comparison-stat">
            <span className="stat-label">Price Range</span>
            <span className="stat-value">
              ${metrics.minPrice.toFixed(0)} - ${metrics.maxPrice.toFixed(0)}
            </span>
          </div>
          <div className="comparison-stat">
            <span className="stat-label">Avg Score</span>
            <span className="stat-value">{metrics.avgScore.toFixed(0)}</span>
          </div>
          <div className="comparison-stat">
            <span className="stat-label">Best Match</span>
            <span className="stat-value highlight">{metrics.bestScore.toFixed(0)}</span>
          </div>
        </div>
      )}

      {selectedProducts.length < 2 && (
        <div className="comparison-hint">
          Select {2 - selectedProducts.length} more product{selectedProducts.length === 1 ? "" : "s"} to compare
        </div>
      )}
    </div>
  );
};

export default ComparisonPanel;
