import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
  plugins: [react()],
  base: "./",  // Use relative paths for Streamlit component compatibility
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false
  }
});
