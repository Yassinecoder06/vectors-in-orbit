import { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';

interface CameraFocusProps {
  focusPosition: [number, number, number] | null;
  orbitControlsRef: React.RefObject<{
    target: THREE.Vector3;
    update: () => void;
    object: THREE.Camera;
  } | null>;
  onFocusComplete?: () => void;
  duration?: number;
}

export default function CameraFocus({
  focusPosition,
  orbitControlsRef,
  onFocusComplete,
  duration = 1500,
}: CameraFocusProps) {
  const { camera } = useThree();
  const animationRef = useRef<number | null>(null);
  const prevFocusRef = useRef<string | null>(null);

  useEffect(() => {
    const controls = orbitControlsRef.current;
    const focusKey = focusPosition ? focusPosition.join(',') : null;

    // Cancel any existing animation
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    // If no focus position or same as before, skip
    if (!focusPosition || !controls || focusKey === prevFocusRef.current) {
      return;
    }

    prevFocusRef.current = focusKey;

    // Get start values
    const startPos = camera.position.clone();
    const startTarget = controls.target.clone();

    // Calculate end values - position camera offset from focus point (zoomed out to show image + text)
    const cameraOffset = 28;
    const cameraHeight = Math.max(focusPosition[1] + 22, 30);
    
    const endPos = new THREE.Vector3(
      focusPosition[0] + cameraOffset,
      cameraHeight,
      focusPosition[2] + cameraOffset
    );
    
    const endTarget = new THREE.Vector3(
      focusPosition[0],
      Math.max(focusPosition[1] + 4, 4),
      focusPosition[2]
    );

    // Animation setup
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Cubic ease-out: eased = 1 - Math.pow(1 - progress, 3)
      const eased = 1 - Math.pow(1 - progress, 3);

      // Interpolate position and target
      camera.position.lerpVectors(startPos, endPos, eased);
      controls.target.lerpVectors(startTarget, endTarget, eased);

      // Update controls
      controls.update();

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      } else {
        animationRef.current = null;
        onFocusComplete?.();
      }
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [focusPosition, orbitControlsRef, camera, duration, onFocusComplete]);

  // Handle escape key to cancel focus
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return null;
}
