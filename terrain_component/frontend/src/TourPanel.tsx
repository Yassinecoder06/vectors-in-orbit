import type { TerrainPoint, Highlight } from './types';

interface TourPanelProps {
  tourActive: boolean;
  currentStep: number;
  totalSteps: number;
  currentProduct: TerrainPoint | null;
  onNext: () => void;
  onPrevious: () => void;
  onStart: () => void;
  onEnd: () => void;
  highlights: Highlight[];
}

// Function to format description text with markdown-like support
function formatDescription(text: string): JSX.Element[] {
  const elements: JSX.Element[] = [];
  
  // Split by double newlines or **headers**
  const lines = text.split(/\n+/);
  let currentList: string[] = [];
  let listKey = 0;
  
  const flushList = () => {
    if (currentList.length > 0) {
      elements.push(
        <ul key={`list-${listKey++}`} className="tour-desc-list">
          {currentList.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>
      );
      currentList = [];
    }
  };
  
  lines.forEach((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    
    // Check for **Header** pattern
    const headerMatch = trimmed.match(/^\*\*(.+?)\*\*$/);
    if (headerMatch) {
      flushList();
      elements.push(
        <h5 key={`h-${index}`} className="tour-desc-header">{headerMatch[1]}</h5>
      );
      return;
    }
    
    // Check for bullet point (- item)
    const bulletMatch = trimmed.match(/^[-•]\s*(.+)$/);
    if (bulletMatch) {
      currentList.push(bulletMatch[1]);
      return;
    }
    
    // Check for inline bold **text**
    if (trimmed.includes('**')) {
      flushList();
      const parts = trimmed.split(/\*\*(.+?)\*\*/g);
      elements.push(
        <p key={`p-${index}`} className="tour-desc-text">
          {parts.map((part, i) => 
            i % 2 === 1 ? <strong key={i}>{part}</strong> : part
          )}
        </p>
      );
      return;
    }
    
    // Regular paragraph
    flushList();
    elements.push(
      <p key={`p-${index}`} className="tour-desc-text">{trimmed}</p>
    );
  });
  
  flushList();
  return elements;
}

export default function TourPanel({
  tourActive,
  currentStep,
  totalSteps,
  currentProduct,
  onNext,
  onPrevious,
  onStart,
  onEnd,
  highlights,
}: TourPanelProps) {
  if (!highlights.length) {
    return null;
  }

  // Tour not started - show start button
  if (!tourActive) {
    return (
      <div className="tour-panel">
        <div className="tour-intro">
          <h3>🏔️ Guided Tour</h3>
          <p>Explore {highlights.length} curated products with our guided tour</p>
          <button className="tour-btn tour-btn-start" onClick={onStart}>
            Start Tour
          </button>
        </div>
      </div>
    );
  }

  // Tour intro (step 0)
  if (currentStep === 0) {
    return (
      <div className="tour-panel tour-active">
        <div className="tour-header">
          <span className="tour-badge">Tour</span>
          <button className="tour-close" onClick={onEnd} title="Cancel Tour">✕</button>
        </div>
        <div className="tour-content">
          <h3>Welcome to the Tour! 👋</h3>
          <p>We'll guide you through {totalSteps} hand-picked products.</p>
          <p>Use the navigation buttons or press <kbd>→</kbd> to continue.</p>
        </div>
        <div className="tour-nav">
          <button className="tour-btn tour-btn-cancel" onClick={onEnd}>Cancel</button>
          <span className="tour-progress">Intro</span>
          <button className="tour-btn tour-btn-primary" onClick={onNext}>Next →</button>
        </div>
      </div>
    );
  }

  // Tour outro (after last product)
  if (currentStep > totalSteps) {
    return (
      <div className="tour-panel tour-active">
        <div className="tour-header">
          <span className="tour-badge">Tour Complete</span>
          <button className="tour-close" onClick={onEnd}>✕</button>
        </div>
        <div className="tour-content">
          <h3>Tour Complete! 🎉</h3>
          <p>You've explored all {totalSteps} products.</p>
          <p>Feel free to continue exploring on your own!</p>
        </div>
        <div className="tour-nav">
          <button className="tour-btn" onClick={onPrevious}>← Back</button>
          <span className="tour-progress">Done</span>
          <button className="tour-btn tour-btn-primary" onClick={onEnd}>Finish</button>
        </div>
      </div>
    );
  }

  // Product step
  return (
    <div className="tour-panel tour-active">
      <div className="tour-header">
        <span className="tour-badge">Tour</span>
        <span className="tour-step">{currentStep} of {totalSteps}</span>
        <button className="tour-close" onClick={onEnd} title="Cancel Tour">✕</button>
      </div>
      {currentProduct && (
        <div className="tour-product">
          {currentProduct.imageUrl && (
            <img 
              src={currentProduct.imageUrl} 
              alt={currentProduct.name}
              className="tour-product-image"
            />
          )}
          <div className="tour-product-info">
            <h4>{currentProduct.name}</h4>
            <div className="tour-product-meta">
              <span className="tour-price">
                ${currentProduct.price.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </span>
              {currentProduct.brand && (
                <span className="tour-brand">{currentProduct.brand}</span>
              )}
            </div>
            {currentProduct.description && (
              <div className="tour-description">
                {formatDescription(currentProduct.description)}
              </div>
            )}
            {currentProduct.category && (
              <span className="tour-category">{currentProduct.category}</span>
            )}
          </div>
        </div>
      )}
      <div className="tour-nav">
        <button 
          className="tour-btn" 
          onClick={onPrevious}
          disabled={currentStep <= 1}
        >
          ← Prev
        </button>
        <div className="tour-dots">
          {Array.from({ length: totalSteps }, (_, i) => (
            <span 
              key={i} 
              className={`tour-dot ${i + 1 === currentStep ? 'active' : ''}`}
            />
          ))}
        </div>
        <button className="tour-btn tour-btn-primary" onClick={onNext}>
          {currentStep === totalSteps ? 'Finish' : 'Next →'}
        </button>
      </div>
    </div>
  );
}
