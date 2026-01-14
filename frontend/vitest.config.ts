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
        'src/lib/**/*.ts',
        'src/components/ui/button.tsx',
        'src/components/ui/card.tsx',
        'src/components/ui/input.tsx',
        'src/components/ui/label.tsx',
        'src/components/dashboard/**/*.tsx',
        'src/app/providers.tsx',
      ],
      exclude: [
        'src/test/**',
        'src/**/*.test.{ts,tsx}',
        'src/**/*.spec.{ts,tsx}',
        'src/**/*.d.ts',
        'src/lib/db/index.ts',
        'src/lib/stripe.ts',
        'src/lib/supabase/**',
      ],
      thresholds: {
        lines: 75,
        functions: 70,
        branches: 60,
        statements: 75,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
