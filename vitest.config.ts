import { defineConfig, type Plugin } from "vitest/config";
import { readFileSync } from "node:fs";

// `?raw` imports (the prompt files) as plain strings, as Vite does in the app.
const raw: Plugin = {
  name: "raw-text",
  load(id) {
    if (id.endsWith("?raw")) return `export default ${JSON.stringify(readFileSync(id.slice(0, -4), "utf8"))};`;
  }
};

// Orchestrator tests are plain TypeScript with scripted model replies; no Workers runtime needed.
export default defineConfig({
  plugins: [raw],
  test: { include: ["tests/**/*.test.ts"], environment: "node" }
});
