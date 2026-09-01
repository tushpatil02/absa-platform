import react from "@vitejs/plugin-react";
// defineConfig comes from vitest/config, not vite, so the `test` block below is
// typed. A `/// <reference types="vitest/config" />` would also work but only if
// it stays the first line of the file, which import sorting does not guarantee —
// it silently drifted below the imports and broke `npm run build`.
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy keeps the browser same-origin in development, so CORS behaviour
    // matches production instead of only working locally.
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: { outDir: "dist", sourcemap: true },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
