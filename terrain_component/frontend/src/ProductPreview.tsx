import type { TerrainPoint } from "./types";

interface ProductPreviewProps {
  product: TerrainPoint | null;
  position?: { x: number; y: number };
  onClose?: () => void;
}

const ProductPreview = ({ product, position, onClose }: ProductPreviewProps) => {
  if (!product) return null;

  const score = Math.round(product.score * 100);
  const scoreColor = score >= 80 ? "#10b981" : score >= 60 ? "#f59e0b" : "#ef4444";

  return (
    <div
      className="product-preview"
      style={{
        left: position?.x ?? "50%",
        top: position?.y ?? "50%",
      }}
    >
      <button className="preview-close" onClick={onClose}>
        ×
      </button>
      
      {product.imageUrl && (
        <div className="preview-image-container">
          <img
            src={product.imageUrl}
            alt={product.name}
            className="preview-image"
            onError={(e) => {
              (e.target as HTMLImageElement).src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'%3E%3Crect fill='%23374151' width='100' height='100'/%3E%3Ctext fill='%239ca3af' x='50' y='50' text-anchor='middle' dy='.3em' font-size='12'%3ENo Image%3C/text%3E%3C/svg%3E";
            }}
          />
          <div className="preview-score-badge" style={{ backgroundColor: scoreColor }}>
            {score}
          </div>
        </div>
      )}

      <div className="preview-content">
        <h4 className="preview-name">{product.name}</h4>
        <div className="preview-price">${product.price.toFixed(2)}</div>
        <div className="preview-meta">
          <span className="preview-brand">{product.brand}</span>
          <span className="preview-category">{product.category}</span>
        </div>
        {product.description && (
          <p className="preview-description">
            {product.description.slice(0, 100)}
            {product.description.length > 100 ? "..." : ""}
          </p>
        )}
        <div className="preview-rank">
          Rank #{product.rank ?? "—"}
        </div>
      </div>
    </div>
  );
};

export default ProductPreview;
