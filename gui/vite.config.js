import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      "/ws": {
        target: "ws://127.0.0.1:8420",
        ws: true,
      },
      "/files": {
        target: "http://127.0.0.1:8420",
      },
    },
  },
});
