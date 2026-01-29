import { useMemo, useState } from "react";
import type { TerrainPoint } from "./types";

interface FilterPanelProps {
  points: TerrainPoint[];
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
  visible?: boolean;
}

export interface FilterState {
  priceRange: [number, number];
  scoreThreshold: number;
  categories: Set<string>;
  brands: Set<string>;
}

export const getDefaultFilters = (points: TerrainPoint[]): FilterState => {
  const prices = points.map((p) => p.price);
  const categories = new Set(points.map((p) => p.category));
  const brands = new Set(points.map((p) => p.brand));
  
  return {
    priceRange: [Math.min(...prices), Math.max(...prices)],
    scoreThreshold: 0,
    categories,
    brands,
  };
};

export const applyFilters = (points: TerrainPoint[], filters: FilterState): TerrainPoint[] => {
  return points.filter((point) => {
    // Price filter
    if (point.price < filters.priceRange[0] || point.price > filters.priceRange[1]) {
      return false;
    }
    // Score filter
    if (point.score * 100 < filters.scoreThreshold) {
      return false;
    }
    // Category filter
    if (filters.categories.size > 0 && !filters.categories.has(point.category)) {
      return false;
    }
    // Brand filter
    if (filters.brands.size > 0 && !filters.brands.has(point.brand)) {
      return false;
    }
    return true;
  });
};

const FilterPanel = ({ points, filters, onFiltersChange, visible = true }: FilterPanelProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Get unique categories and brands
  const { allCategories, allBrands, priceMin, priceMax } = useMemo(() => {
    const categories = [...new Set(points.map((p) => p.category))].sort();
    const brands = [...new Set(points.map((p) => p.brand))].sort();
    const prices = points.map((p) => p.price);
    return {
      allCategories: categories,
      allBrands: brands,
      priceMin: Math.min(...prices),
      priceMax: Math.max(...prices),
    };
  }, [points]);

  const handlePriceMinChange = (value: number) => {
    onFiltersChange({
      ...filters,
      priceRange: [value, filters.priceRange[1]],
    });
  };

  const handlePriceMaxChange = (value: number) => {
    onFiltersChange({
      ...filters,
      priceRange: [filters.priceRange[0], value],
    });
  };

  const handleScoreChange = (value: number) => {
    onFiltersChange({
      ...filters,
      scoreThreshold: value,
    });
  };

  const toggleCategory = (category: string) => {
    const newCategories = new Set(filters.categories);
    if (newCategories.has(category)) {
      newCategories.delete(category);
    } else {
      newCategories.add(category);
    }
    // If all are unchecked, select all (show all)
    if (newCategories.size === 0) {
      allCategories.forEach((c) => newCategories.add(c));
    }
    onFiltersChange({ ...filters, categories: newCategories });
  };

  const toggleBrand = (brand: string) => {
    const newBrands = new Set(filters.brands);
    if (newBrands.has(brand)) {
      newBrands.delete(brand);
    } else {
      newBrands.add(brand);
    }
    // If all are unchecked, select all
    if (newBrands.size === 0) {
      allBrands.forEach((b) => newBrands.add(b));
    }
    onFiltersChange({ ...filters, brands: newBrands });
  };

  const resetFilters = () => {
    onFiltersChange(getDefaultFilters(points));
  };

  const filteredCount = applyFilters(points, filters).length;

  if (!visible) return null;

  return (
    <div className={`filter-panel ${isExpanded ? "expanded" : "collapsed"}`}>
      <div className="filter-header" onClick={() => setIsExpanded(!isExpanded)}>
        <span className="filter-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
          </svg>
        </span>
        <span className="filter-title">Filters</span>
        <span className="filter-count">{filteredCount}/{points.length}</span>
        <span className={`filter-chevron ${isExpanded ? "up" : "down"}`}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </span>
      </div>
      
      {isExpanded && (
        <div className="filter-content">
          {/* Price Range */}
          <div className="filter-section">
            <label className="filter-label">Price Range</label>
            <div className="filter-range">
              <input
                type="range"
                min={priceMin}
                max={priceMax}
                value={filters.priceRange[0]}
                onChange={(e) => handlePriceMinChange(Number(e.target.value))}
                className="range-input"
              />
              <input
                type="range"
                min={priceMin}
                max={priceMax}
                value={filters.priceRange[1]}
                onChange={(e) => handlePriceMaxChange(Number(e.target.value))}
                className="range-input"
              />
            </div>
            <div className="filter-range-values">
              <span>${filters.priceRange[0].toFixed(0)}</span>
              <span>${filters.priceRange[1].toFixed(0)}</span>
            </div>
          </div>

          {/* Score Threshold */}
          <div className="filter-section">
            <label className="filter-label">Min Score: {filters.scoreThreshold}</label>
            <input
              type="range"
              min={0}
              max={100}
              value={filters.scoreThreshold}
              onChange={(e) => handleScoreChange(Number(e.target.value))}
              className="range-input score-range"
            />
          </div>

          {/* Categories */}
          <div className="filter-section">
            <label className="filter-label">Categories</label>
            <div className="filter-checkboxes">
              {allCategories.slice(0, 5).map((category) => (
                <label key={category} className="filter-checkbox">
                  <input
                    type="checkbox"
                    checked={filters.categories.has(category)}
                    onChange={() => toggleCategory(category)}
                  />
                  <span>{category}</span>
                </label>
              ))}
              {allCategories.length > 5 && (
                <span className="filter-more">+{allCategories.length - 5} more</span>
              )}
            </div>
          </div>

          {/* Brands */}
          <div className="filter-section">
            <label className="filter-label">Brands</label>
            <div className="filter-checkboxes">
              {allBrands.slice(0, 5).map((brand) => (
                <label key={brand} className="filter-checkbox">
                  <input
                    type="checkbox"
                    checked={filters.brands.has(brand)}
                    onChange={() => toggleBrand(brand)}
                  />
                  <span>{brand}</span>
                </label>
              ))}
              {allBrands.length > 5 && (
                <span className="filter-more">+{allBrands.length - 5} more</span>
              )}
            </div>
          </div>

          {/* Reset Button */}
          <button className="filter-reset" onClick={resetFilters}>
            Reset Filters
          </button>
        </div>
      )}
    </div>
  );
};

export default FilterPanel;
