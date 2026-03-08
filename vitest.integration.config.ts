import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    pool: 'threads',
    singleThread: true,
    include: ['test/integration.test.ts'],
    testTimeout: 120_000,
  },
});
