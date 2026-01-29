# 🏔️ Terrain Visualizer Improvement Plan

## Based on Analysis of: [Vector Vintage](https://github.com/kanungle/vector-vintage-public)

---

## 📋 Executive Summary

Vector Vintage is a hackathon project that visualizes e-commerce products on a 3D terrain where:
- **Mountains** = Product categories (height based on # of search results)
- **Hills** = Subcategories surrounding mountains
- Products are positioned using **UMAP embeddings** of categories
- A guided **Q-Bert mascot tour** walks users through curated products
- Uses **Qdrant** for vector search + **Neo4J** for relationship filtering

Your current project already has many similar concepts. Below are features we can adopt to enhance your terrain explorer.

---

## 🎯 Priority 1: WASD Keyboard Navigation (HIGH IMPACT)

### What They Did
They have a dedicated `KeyboardControls.tsx` component that:
- Uses `requestAnimationFrame` for smooth movement (NOT `useFrame`)
- Tracks pressed keys with `useRef<Set<string>>`
- Calculates forward/right vectors from camera orientation
- Moves both camera and OrbitControls target together

### Implementation Plan
```
1. Create separate KeyboardControls component that uses requestAnimationFrame
2. Track keys in a Set<string> ref
3. Get camera direction vectors (forward.y = 0 to keep horizontal)
4. Move camera.position AND orbitControls.target together
5. Support: W/S (forward/back), A/D (strafe), Q/E or PageUp/PageDown (altitude)
```

### Why This Will Work (vs previous failures)
Previous attempts used `useFrame` from `@react-three/fiber` which caused "Component Error" in the Streamlit wrapper. Vector Vintage uses **native `requestAnimationFrame`** + window event listeners instead, which is browser-native and won't conflict with the component system.

---

## 🎯 Priority 2: Smooth Camera Focus Animation

### What They Did
Their `CameraController` function smoothly animates camera to focus on products:
- Cubic ease-out animation over 1.5 seconds
- Uses `lerpVectors` for smooth position interpolation
- Keeps camera at minimum height (never below 15)
- Auto-clears focus after 5 seconds with timeout

### Implementation Plan
```
1. Store startPos/startTarget and endPos/endTarget
2. Use requestAnimationFrame (not useFrame) for animation loop
3. Apply cubic ease-out: eased = 1 - Math.pow(1 - progress, 3)
4. Lerp camera.position and controls.target
5. Add escape key handler to cancel focus
```

---

## 🎯 Priority 3: Terrain Shader Material with Textures

### What They Did
Created a custom `TerrainMaterial` class extending `THREE.ShaderMaterial`:
- Uses vertex shader to pass height/UV/normals
- Fragment shader applies grass/rock/snow textures based on height
- `snowShift` uniform for dynamic weather effects
- Circular clip radius for playable area boundary

### Implementation Plan
```
1. Create TerrainMaterial class with custom GLSL shaders
2. Add texture uniforms for grass/rock/snow
3. Height-based texture blending in fragment shader
4. Add snowShift uniform for weather control
5. Keep circular boundary with discard in shader
```

### Assets Needed
- `/textures/diffuse_baseColor.jpeg` (grass/dirt)
- `/textures/diffuse_normal.png` (normal map)

---

## 🎯 Priority 4: Environmental Decorations (ForestLandmarks)

### What They Did
Added immersive environment elements:
- **Trees** - Placed on grassy slopes, avoiding water
- **Alpine Cows** - Grazing on gentle slopes
- **Settlements** - In valley/flat areas
- **Campfires** - With GLB models
- **Trail Signs** - Interactive clickable elements

### Implementation Plan
```
1. Create placement utility functions:
   - findValidPosition() - avoid water, respect slope constraints
   - projectToSurface() - snap objects to terrain
   - generateHeightGrid() - pre-calculate terrain heights

2. Tree Component:
   - Cone geometry for trunk + sphere for foliage
   - Shared materials for performance
   - Random scale variation

3. Placement Rules:
   - minRadius/maxRadius from center
   - maxSlope constraint
   - avoidWater with buffer radius
   - minHeight threshold
```

---

## 🎯 Priority 5: Dynamic Mountain/Hill Labels

### What They Did
`MountainLabels.tsx` with:
- Billboard Text components that face camera
- Labels positioned above terrain features
- Distance-gated visibility for hills (fade based on camera distance)
- Clean name formatting (strip "Mount", "Peak", "Summit", etc.)

### Implementation Plan
```
1. Use Billboard + Text from @react-three/drei
2. Position labels at terrain height + offset
3. Use useFrame to make labels lookAt camera (already doing this)
4. Add distance-based opacity for performance
5. Format names by removing common suffixes
```

---

## 🎯 Priority 6: Product Icons/Markers (IconMarker)

### What They Did
Replaced plain product images with **category-specific 3D icons**:
- 40+ subcategory icons (AI-generated via Sloyd.ai)
- Icons positioned on terrain near their category hills
- Highlight state on selection
- Hover tooltip with product info

### Implementation Plan (Simplified)
```
1. Keep current image-based approach (faster for hackathon)
2. Add highlight ring/glow for selected products
3. Improve hover state with scale animation
4. Add relevance score indicator (distance from search query)
```

---

## 🎯 Priority 7: Control Indicators UI

### What They Did
Overlay showing keyboard controls:
- WASD keys with visual feedback when pressed
- Arrow key alternatives
- Q/E for altitude
- Scroll for zoom info

### Implementation Plan
```
1. Create ControlIndicators component in Streamlit sidebar or overlay
2. Show WASD diagram with active key highlighting
3. Add control hints: "Scroll to zoom", "Click product for details"
```

---

## 🎯 Priority 8: Query Path Visualization

### What They Did
`QueryPath.tsx` draws animated lines between products:
- Uses `Line` from @react-three/drei
- Animated dash offset for "flowing" effect
- Connects products in search result order

### Implementation Plan
```
1. Calculate path points from product positions
2. Use Line component with dashed material
3. Animate dashOffset using useFrame (or requestAnimationFrame)
4. Show path when products are filtered/searched
```

---

## 🎯 Priority 9: Product Tour System

### What They Did
"Q-Bert" mascot that guides users through products:
- LLM selects 7 highlighted products
- Tour navigation: Previous/Next buttons
- Camera auto-focuses on current tour product
- Product card shows tour step indicator

### Implementation Plan
```
1. Add tour mode state (active/inactive)
2. Store tour products array + current step
3. On step change: focus camera on product position
4. Add Next/Previous buttons in UI
5. Show step indicator: "Product 3 of 7"
```

---

## 🚀 Implementation Order (Recommended)

| Order | Feature | Effort | Impact |
|-------|---------|--------|--------|
| 1 | WASD Navigation (requestAnimationFrame) | Medium | HIGH |
| 2 | Smooth Camera Animation | Low | Medium |
| 3 | Control Indicators UI | Low | Medium |
| 4 | Terrain Textures/Shader | High | HIGH |
| 5 | Environmental Decorations | Medium | Medium |
| 6 | Product Tour System | Medium | HIGH |
| 7 | Query Path Animation | Low | Low |
| 8 | Mountain/Hill Labels (improve) | Low | Low |

---

## ⚠️ Important Notes

### What NOT to do (based on previous failures):
1. **DO NOT use `useFrame` from @react-three/fiber** - causes Component Error in Streamlit
2. **DO NOT add `ref` to OrbitControls directly** - breaks the component
3. **DO NOT import controlsRef patterns** - use native browser APIs instead

### What TO do:
1. Use `window.addEventListener('keydown/keyup')` for keyboard
2. Use `requestAnimationFrame` for animation loops
3. Keep the Streamlit component wrapper simple
4. Test each change incrementally

---

## 📁 Files to Create/Modify

### New Files:
- `terrain_component/frontend/src/components/KeyboardControls.tsx`
- `terrain_component/frontend/src/components/ControlIndicators.tsx`
- `terrain_component/frontend/src/utils/terrainGrid.ts`
- `terrain_component/frontend/public/textures/` (texture assets)

### Files to Modify:
- `terrain_component/frontend/src/App.tsx` - integrate new components
- `terrain_component/frontend/src/styles.css` - control indicator styles
- `app.py` - add tour mode controls in Streamlit

---

## 🎨 Visual Improvements Summary

| Current State | Proposed Improvement |
|---------------|---------------------|
| White terrain | Textured terrain (grass/rock/snow) |
| Static camera | WASD movement + smooth focus |
| Plain labels | Billboard labels with distance fade |
| No environment | Trees, settlements, decorations |
| No path viz | Animated search result path |
| No tour mode | Guided product tour system |

---

## ✅ Ready to Implement?

Let me know which features you'd like to implement first and I'll start coding them. I recommend starting with **WASD Navigation** since it was requested multiple times and the new approach (using requestAnimationFrame instead of useFrame) should work without causing Component Errors.
