# 🏔️ Terrain Visualization - Feature Roadmap

## ✅ Completed Features
- [x] 3D mountain terrain based on product scores
- [x] Dynamic height scaling (score → mountain elevation)
- [x] Product labels with match score badges (0-100)
- [x] Query path connecting products by rank order
- [x] Guided tour system with keyboard navigation
- [x] Fullscreen toggle button
- [x] River flowing around mountain perimeter
- [x] Low-poly trees and cloud decorations
- [x] Alpine color gradient (green → rock → snow)
- [x] WASD keyboard camera controls
- [x] Click-to-focus on products
- [x] Product selection with Streamlit integration

---

## 🎯 High Priority Features

### 1. Product Image Previews
- Show product thumbnail on hover/focus
- Mini image card floating near the product marker
- Lazy load images for performance

### 2. Category/Brand Color Coding
- Color-code product markers by category or brand
- Add a legend overlay showing color meanings
- Toggle between score-based and category-based coloring

### 3. Filter Controls Overlay
- Price range slider
- Category checkboxes
- Brand filter dropdown
- Score threshold slider
- Show/hide filtered products with fade animation

### 4. Mini-Map Navigation
- Small 2D overhead map in corner
- Show all products as dots
- Click to teleport camera
- Highlight current viewport area

### 5. Comparison Mode
- Select 2-3 products to compare side-by-side
- Show comparison panel with specs
- Highlight selected products on terrain

---

## 🚀 Medium Priority Features

### 6. Animation Improvements
- Smooth product entry animations on load
- Pulsing effect for top-ranked products
- Particle effects at mountain peak
- Animated path "flow" showing rank direction

### 7. Search Within Visualization
- Text search box to highlight matching products
- Zoom to search result
- Highlight search matches with glow effect

### 8. Time-of-Day / Weather Effects
- Day/night cycle toggle
- Sunrise/sunset lighting moods
- Weather variations (clear, cloudy, misty)
- Affects atmosphere and mood

### 9. Product Clustering Visualization
- Group similar products into visible clusters
- Show cluster boundaries on terrain
- Expand/collapse clusters on click

### 10. Score Breakdown Tooltip
- On hover, show pie chart of score components
- Semantic, Affordability, Preference, Collaborative, Popularity
- Visual explanation of why product ranked where it is

---

## 💡 Nice-to-Have Features

### 11. Export & Share
- Screenshot button (capture current view)
- Share URL with camera position encoded
- Export product list as CSV

### 12. Accessibility Improvements
- Screen reader support for product info
- High contrast mode
- Keyboard-only navigation improvements
- Reduce motion option

### 13. Sound Design
- Ambient mountain/nature sounds
- Click/selection sound effects
- Tour narration audio option

### 14. Performance Optimizations
- Level-of-detail (LOD) for distant products
- Frustum culling for off-screen elements
- Instanced rendering for trees/decorations

### 15. Mobile/Touch Support
- Pinch to zoom
- Swipe to rotate
- Touch-friendly product selection
- Responsive UI panels

### 16. AR Mode (Stretch Goal)
- View terrain in augmented reality
- Place mountain on table via WebXR
- Walk around the visualization

---

## 🔧 Technical Improvements

### Code Quality
- [ ] Add unit tests for terrain generation
- [ ] TypeScript strict mode compliance
- [ ] Component documentation with Storybook
- [ ] Performance profiling and optimization

### Infrastructure
- [ ] Caching layer for terrain geometry
- [ ] WebWorker for heavy computations
- [ ] Progressive loading for large product sets
- [ ] Error boundaries and fallback UI

---

## 📊 Analytics Features

### User Behavior Tracking
- Heatmap of most-viewed products
- Time spent at each product
- Tour completion rate
- Most common camera paths

### Recommendation Insights
- A/B testing different scoring weights
- Conversion tracking from visualization
- User preference learning from interactions

---

## 🎨 Visual Polish

- [ ] Better snow cap textures
- [ ] Animated water/river shader
- [ ] Fog/atmosphere depth effect
- [ ] Lens flare from sun
- [ ] Product marker redesign (3D models?)
- [ ] Custom fonts for labels
- [ ] Loading screen with progress

---

*Last updated: January 30, 2026*
