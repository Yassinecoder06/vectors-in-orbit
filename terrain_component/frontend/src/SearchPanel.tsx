import { useState, useCallback, useMemo } from "react";
import type { TerrainPoint } from "./types";

interface SearchPanelProps {
  points: TerrainPoint[];
  onSearchResult: (matchedIds: Set<string>) => void;
  onFocusProduct: (point: TerrainPoint) => void;
  visible?: boolean;
}

const SearchPanel = ({ points, onSearchResult, onFocusProduct, visible = true }: SearchPanelProps) => {
  const [query, setQuery] = useState("");
  const [isExpanded, setIsExpanded] = useState(false);

  // Search results
  const searchResults = useMemo(() => {
    if (!query.trim()) return [];
    
    const lowerQuery = query.toLowerCase();
    return points.filter((point) => 
      point.name.toLowerCase().includes(lowerQuery) ||
      point.brand.toLowerCase().includes(lowerQuery) ||
      point.category.toLowerCase().includes(lowerQuery) ||
      (point.description?.toLowerCase().includes(lowerQuery))
    ).slice(0, 8); // Max 8 results
  }, [points, query]);

  // Update matched IDs when search changes
  const handleSearch = useCallback((value: string) => {
    setQuery(value);
    if (!value.trim()) {
      onSearchResult(new Set());
    } else {
      const lowerQuery = value.toLowerCase();
      const matchedIds = new Set(
        points
          .filter((point) => 
            point.name.toLowerCase().includes(lowerQuery) ||
            point.brand.toLowerCase().includes(lowerQuery) ||
            point.category.toLowerCase().includes(lowerQuery) ||
            (point.description?.toLowerCase().includes(lowerQuery))
          )
          .map((p) => p.id)
      );
      onSearchResult(matchedIds);
    }
  }, [points, onSearchResult]);

  const handleResultClick = useCallback((point: TerrainPoint) => {
    onFocusProduct(point);
    setIsExpanded(false);
  }, [onFocusProduct]);

  const clearSearch = useCallback(() => {
    setQuery("");
    onSearchResult(new Set());
  }, [onSearchResult]);

  if (!visible) return null;

  return (
    <div className={`search-panel ${isExpanded ? "expanded" : ""}`}>
      <div className="search-input-wrapper">
        <svg className="search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
        <input
          type="text"
          className="search-input"
          placeholder="Search products..."
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          onFocus={() => setIsExpanded(true)}
        />
        {query && (
          <button className="search-clear" onClick={clearSearch}>
            ×
          </button>
        )}
      </div>

      {isExpanded && searchResults.length > 0 && (
        <div className="search-results">
          {searchResults.map((result) => (
            <div
              key={result.id}
              className="search-result-item"
              onClick={() => handleResultClick(result)}
            >
              <div className="search-result-name">{result.name}</div>
              <div className="search-result-meta">
                <span className="search-result-brand">{result.brand}</span>
                <span className="search-result-price">${result.price.toFixed(0)}</span>
                <span className="search-result-score">{Math.round(result.score * 100)}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {isExpanded && query && searchResults.length === 0 && (
        <div className="search-no-results">
          No products found matching "{query}"
        </div>
      )}

      {query && (
        <div className="search-match-count">
          {searchResults.length} match{searchResults.length !== 1 ? "es" : ""} highlighted
        </div>
      )}
    </div>
  );
};

export default SearchPanel;
