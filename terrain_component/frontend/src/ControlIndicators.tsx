import { useEffect, useState } from 'react';

interface ControlIndicatorsProps {
  visible?: boolean;
}

export default function ControlIndicators({ visible = true }: ControlIndicatorsProps) {
  const [activeKeys, setActiveKeys] = useState<Set<string>>(new Set());

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      setActiveKeys((prev) => new Set(prev).add(key));
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      setActiveKeys((prev) => {
        const next = new Set(prev);
        next.delete(key);
        return next;
      });
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  if (!visible) return null;

  const isActive = (keys: string[]) => keys.some((k) => activeKeys.has(k));

  return (
    <div className="control-indicators">
      <div className="control-section">
        <div className="control-title">Navigate</div>
        <div className="wasd-grid">
          <div className="key-row">
            <span className={`key ${isActive(['w', 'arrowup']) ? 'active' : ''}`}>W</span>
          </div>
          <div className="key-row">
            <span className={`key ${isActive(['a', 'arrowleft']) ? 'active' : ''}`}>A</span>
            <span className={`key ${isActive(['s', 'arrowdown']) ? 'active' : ''}`}>S</span>
            <span className={`key ${isActive(['d', 'arrowright']) ? 'active' : ''}`}>D</span>
          </div>
        </div>
        <div className="control-hint">or Arrow Keys</div>
      </div>

      <div className="control-section">
        <div className="control-title">Altitude</div>
        <div className="altitude-keys">
          <span className={`key ${isActive(['q', 'pageup']) ? 'active' : ''}`}>Q</span>
          <span className="key-label">Up</span>
          <span className={`key ${isActive(['e', 'pagedown']) ? 'active' : ''}`}>E</span>
          <span className="key-label">Down</span>
        </div>
      </div>

      <div className="control-section">
        <div className="control-title">Mouse</div>
        <div className="mouse-hints">
          <span>🖱️ Drag to rotate</span>
          <span>⚙️ Scroll to zoom</span>
          <span>👆 Click product for details</span>
        </div>
      </div>
    </div>
  );
}
