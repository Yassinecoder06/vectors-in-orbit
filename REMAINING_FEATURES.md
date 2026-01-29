# 🚀 Remaining Features to Implement

## ✅ Completed
- [x] WASD Keyboard Navigation (using requestAnimationFrame)
- [x] Camera Focus Animation (smooth cubic ease-out)
- [x] Control Indicators UI (WASD overlay with active key highlighting)
- [x] Product Tour System (guided tour with navigation)

---

## 🎯 Feature 1: Smooth Camera Focus Animation

### Description
When clicking on a product, smoothly animate the camera to focus on it instead of jumping instantly.

### Implementation
```tsx
// CameraController component with cubic ease-out animation
- Store startPos/startTarget and endPos/endTarget
- Use requestAnimationFrame for animation loop (1.5s duration)
- Apply easing: eased = 1 - Math.pow(1 - progress, 3)
- Lerp camera.position and controls.target
- Auto-clear focus after 5 seconds
- Escape key cancels focus
```

### Effort: Low | Impact: Medium

---

## 🎯 Feature 2: Terrain Shader with Textures

### Description
Replace vertex colors with a proper shader material that blends grass/rock/snow textures based on height.

### Implementation
```tsx
// Custom TerrainMaterial extending THREE.ShaderMaterial
- Vertex shader: pass height, UV, normals, world position
- Fragment shader: 
  - Sample grass texture for low heights
  - Blend to rock for mid heights
  - Snow caps on peaks
- Add snowShift uniform for weather control
- Circular clip radius for playable area boundary
```

### Assets Needed
- `/textures/grass_baseColor.jpg`
- `/textures/rock_baseColor.jpg`
- `/textures/snow_baseColor.jpg`
- `/textures/terrain_normal.png`

### Effort: High | Impact: High

---

## 🎯 Feature 3: Environmental Decorations

### Description
Add trees, rocks, and other props to make the terrain feel alive.

### Implementation
```tsx
// ForestLandmarks component
1. Tree Component:
   - Cone trunk + layered spheres for foliage
   - Shared materials for performance
   - Random scale/rotation variation

2. Placement Utilities:
   - findValidPosition(constraints, attempts)
   - projectToSurface(x, z, lift) 
   - Avoid water/steep slopes
   - Respect min/max height

3. Prop Types:
   - Trees (clustered in groups)
   - Rocks (scattered on slopes)
   - Grass patches (low areas)
```

### Effort: Medium | Impact: Medium

---

## 🎯 Feature 4: Control Indicators UI

### Description
Overlay showing keyboard controls with visual feedback when keys are pressed.

### Implementation
```tsx
// ControlIndicators component (HTML overlay or Streamlit sidebar)
- WASD key diagram
- Arrow key alternatives
- Q/E altitude controls
- Mouse controls hints
- Active key highlighting
```

### Can be implemented in:
- React (HTML overlay in terrain component)
- Streamlit sidebar (Python)

### Effort: Low | Impact: Medium

---

## 🎯 Feature 5: Product Tour System

### Description
Guided tour that walks users through highlighted/curated products.

### Implementation
```tsx
// Tour Mode State
- tourActive: boolean
- tourProducts: TerrainPoint[]
- currentStep: number (0 = intro, 1-N = products, N+1 = outro)

// Tour Navigation
- Next/Previous buttons
- Step indicator: "Product 3 of 7"
- Auto-focus camera on current product
- Tour card with product details

// Integration
- Streamlit controls tour start/stop
- Camera animates between products
- Highlight current tour product
```

### Effort: Medium | Impact: High

---

## 🎯 Feature 6: Query Path Visualization

### Description
Animated line connecting products in search result order, showing the "journey" through results.

### Implementation
```tsx
// QueryPath component
- Calculate path points from product positions
- Use Line from @react-three/drei
- Dashed material with animated dashOffset
- Color gradient along path (optional)
- Show/hide based on search state
```

### Effort: Low | Impact: Low

---

## 🎯 Feature 7: Mountain/Hill Category Labels

### Description
Floating labels above terrain regions showing category names.

### Implementation
```tsx
// CategoryLabels component
- Billboard + Text for each category
- Position at category centroid + height offset
- Distance-based opacity (fade when far)
- Clean name formatting
- Optional: wooden sign 3D model
```

### Effort: Low | Impact: Low

---

## 📊 Implementation Priority Matrix

| Feature | Effort | Impact | Priority |
|---------|--------|--------|----------|
| Camera Focus Animation | Low | Medium | 1 |
| Control Indicators UI | Low | Medium | 2 |
| Product Tour System | Medium | High | 3 |
| Terrain Textures | High | High | 4 |
| Environmental Decorations | Medium | Medium | 5 |
| Query Path Visualization | Low | Low | 6 |
| Category Labels | Low | Low | 7 |

---

## 🎬 Recommended Implementation Order

### Phase 1: Quick Wins (1-2 hours)
1. ✨ Camera Focus Animation
2. 🎮 Control Indicators UI

### Phase 2: Core Features (2-3 hours)
3. 🚶 Product Tour System

### Phase 3: Visual Polish (3-4 hours)
4. 🏔️ Terrain Textures/Shader
5. 🌲 Environmental Decorations

### Phase 4: Nice-to-Have (1 hour)
6. 📍 Query Path Visualization
7. 🏷️ Category Labels

---

## 💡 Quick Start Commands

When ready to implement, just say:
- "Do camera focus animation"
- "Do control indicators"
- "Do product tour"
- "Do terrain textures"
- "Do environment decorations"
- "Do all remaining features"
