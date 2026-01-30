# UI Refinements Analysis & Plan

## Problem Identified

The component error is caused by changes in `TourPanel.tsx` that were committed in commit `a57084b` (test10). The `formatDescription()` function has potential issues:

### Root Cause
The `formatDescription()` function:
1. Returns `JSX.Element[]` which can be empty if description has no content
2. Uses complex regex parsing that may fail on edge cases
3. The `listKey` variable is mutated inside `flushList()` closure which can cause React key issues

### Files Modified (in commit a57084b)
1. **`terrain_component/frontend/src/TourPanel.tsx`**
   - Added `formatDescription()` function (lines 15-81)
   - Changed description rendering from `<p>` to `<div>` with parsed content
   - Modified navigation buttons

2. **`terrain_component/frontend/src/styles.css`**
   - Added `.tour-description` scrollbar and overflow styles
   - Added `.tour-desc-header`, `.tour-desc-list`, `.tour-desc-text` classes
   - Added `.tour-btn-cancel` styles
   - Added `.tour-nav-left` styles

---

## Refinements Requested by User

### 1. Cancel Tour Mid-Tour
**Goal:** Allow users to cancel/exit the tour at any point during the tour.

**Solution:** 
- The ✕ button in the tour header already calls `onEnd()` which cancels the tour
- This is already working - no additional button needed
- Just ensure the ✕ button is clearly visible and accessible

### 2. Better Product Description Display
**Goal:** Format descriptions with headers, bullet points, and proper structure (as shown in user's screenshots).

**Solution:**
- Parse markdown-like syntax: `**Header**` for bold headers, `- item` for bullets
- Render structured HTML instead of plain text paragraph
- Add scrolling for long descriptions

### 3. Product Images Hide Behind Obstacles (from earlier request)
**Goal:** Product billboard images should be occluded by terrain/mountains when viewed from behind.

**Approach (CAREFUL - this caused the error before):**
- Change `depthTest={false}` to `depthTest={true}` on the `meshBasicMaterial` in `ProductBillboard`
- This enables depth testing so images are properly hidden behind 3D geometry
- Keep `depthWrite={false}` to avoid z-fighting issues

### 4. Product Labels Under UI Panels (from earlier request)
**Goal:** Product labels should appear behind tour panel and comparison panel.

**Approach (CAREFUL - `occlude` prop caused errors):**
- Do NOT use the `occlude` prop on `Html` component - it requires mesh references and causes errors
- Use `zIndexRange={[50, 0]}` on the `Html` component to set z-index below UI panels (which use z-index 100-160)
- Alternative: Use CSS `z-index` on the product-label class itself

---

## Recommended Fix Strategy

### Step 1: Revert TourPanel.tsx to Original
Restore the original simple description rendering to fix the component error.

### Step 2: Re-implement Carefully
If description formatting is still desired:
- Use a simpler approach: CSS `white-space: pre-line` to preserve line breaks
- Or use a proven markdown library like `react-markdown`
- Avoid complex custom parsing functions

### Step 3: Test Z-Index Changes in Isolation
Before combining changes:
- Test `zIndexRange` alone without `occlude`
- Test `depthTest` changes alone
- Only combine after individual testing

---

## Original Files (for reference)

### Original TourPanel.tsx description rendering:
```tsx
{currentProduct.description && (
  <p className="tour-description">{currentProduct.description}</p>
)}
```

### Original ProductBillboard material:
```tsx
<meshBasicMaterial map={texture} transparent side={THREE.DoubleSide} depthTest={false} />
```

### Original ProductLabels Html component:
```tsx
<Html center transform distanceFactor={8} style={{ pointerEvents: "auto" }}>
```

---

## Commands to Revert

To fully revert to the working state before my changes:
```bash
git checkout a57084b~1 -- terrain_component/frontend/src/TourPanel.tsx
git checkout a57084b~1 -- terrain_component/frontend/src/styles.css
```
