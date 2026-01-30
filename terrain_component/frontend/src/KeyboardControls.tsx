import { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';

interface KeyboardControlsProps {
  enabled?: boolean;
  moveSpeed?: number;
  orbitControlsRef: React.RefObject<{
    target: THREE.Vector3;
    update: () => void;
    object: THREE.Camera;
  } | null>;
  onKeyPress?: (key: string) => void;
}

export default function KeyboardControls({
  enabled = true,
  moveSpeed = 1.0,
  orbitControlsRef,
  onKeyPress,
}: KeyboardControlsProps) {
  const { camera } = useThree();
  const keysPressed = useRef<Set<string>>(new Set());
  const animationIdRef = useRef<number | null>(null);

  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input fields
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      const key = e.key.toLowerCase();
      keysPressed.current.add(key);
      onKeyPress?.(key);
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      keysPressed.current.delete(key);
    };

    // Animation loop for smooth movement using requestAnimationFrame (NOT useFrame)
    const animate = () => {
      const keys = keysPressed.current;
      const controls = orbitControlsRef.current;

      if (!controls || keys.size === 0) {
        animationIdRef.current = requestAnimationFrame(animate);
        return;
      }

      // Get camera direction vectors for panning
      const forward = new THREE.Vector3();
      const right = new THREE.Vector3();

      // Get the horizontal direction the camera is looking
      camera.getWorldDirection(forward);
      forward.y = 0; // Keep movement horizontal
      forward.normalize();

      // Get right vector
      right.crossVectors(forward, camera.up).normalize();

      // Move both camera and target together for panning effect
      const movement = new THREE.Vector3();

      // WASD or Arrow keys for horizontal movement
      if (keys.has('w') || keys.has('arrowup')) {
        movement.addScaledVector(forward, moveSpeed);
      }
      if (keys.has('s') || keys.has('arrowdown')) {
        movement.addScaledVector(forward, -moveSpeed);
      }
      if (keys.has('a') || keys.has('arrowleft')) {
        movement.addScaledVector(right, -moveSpeed);
      }
      if (keys.has('d') || keys.has('arrowright')) {
        movement.addScaledVector(right, moveSpeed);
      }

      // Q/E or PageUp/PageDown for vertical movement
      if (keys.has('q') || keys.has('pageup')) {
        movement.y += moveSpeed;
      }
      if (keys.has('e') || keys.has('pagedown')) {
        movement.y -= moveSpeed;
      }

      // Apply movement to both camera and controls target
      if (movement.lengthSq() > 0) {
        camera.position.add(movement);
        controls.target.add(movement);
        controls.update();
      }

      animationIdRef.current = requestAnimationFrame(animate);
    };

    // Start animation loop
    animationIdRef.current = requestAnimationFrame(animate);

    // Add event listeners
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      // Cleanup
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      keysPressed.current.clear();
    };
  }, [enabled, moveSpeed, camera, orbitControlsRef, onKeyPress]);

  return null;
}
