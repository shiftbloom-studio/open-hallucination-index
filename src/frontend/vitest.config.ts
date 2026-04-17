import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['node_modules', 'e2e/**/*'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      reportsDirectory: './coverage',
      include: [
        'src/lib/ohi-types.ts',
        'src/lib/ohi-client.ts',
        'src/lib/ohi-queries.ts',
        'src/lib/sse.ts',
        'src/lib/verify-controller.ts',
        'src/components/verify/**/*.tsx',
        'src/components/common/**/*.tsx',
        'src/components/calibration/**/*.tsx',
        'src/components/status/**/*.tsx',
        'src/app/providers.tsx',
      ],
      exclude: [
        'src/test/**',
        'src/**/*.test.{ts,tsx}',
        'src/**/*.spec.{ts,tsx}',
        'src/**/*.d.ts',
      ],
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
