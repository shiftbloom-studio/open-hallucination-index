# Testing Guide

This project includes comprehensive unit tests and end-to-end (E2E) tests to ensure code quality and reliability.

## Test Stack

- **Unit Tests**: [Vitest](https://vitest.dev/) with [React Testing Library](https://testing-library.com/react)
- **E2E Tests**: [Playwright](https://playwright.dev/)
- **API Mocking**: [MSW (Mock Service Worker)](https://mswjs.io/)
- **Coverage**: Vitest V8 Coverage

## Running Tests

### Unit Tests

```bash
# Run unit tests (watch mode)
npm run test

# Run tests in watch mode
npm run test:watch

# Run tests once (CI mode)
npm run test:run

# Run tests with UI
npm run test:ui

# Run tests with coverage report
npm run test:coverage
```

### E2E Tests

```bash
# Run all E2E tests
npm run test:e2e

# Run E2E tests with UI
npm run test:e2e:ui

# Run E2E tests in headed mode (see the browser)
npm run test:e2e:headed

# Run E2E tests in debug mode
npm run test:e2e:debug
```

### All Tests

```bash
# Run unit tests and E2E tests
npm run test:all
```

## Test Structure

```
src/
├── test/
│   ├── setup.ts              # Vitest setup file
│   ├── test-utils.tsx        # Custom render with providers
│   ├── mocks/
│   │   ├── handlers.ts       # MSW request handlers
│   │   └── server.ts         # MSW server setup
│   └── integration/
│       ├── api.test.ts       # API integration tests
│       └── user-flows.test.tsx
├── lib/
│   └── __tests__/
│       ├── api.test.ts       # API client tests
│       ├── stripe.test.ts    # Stripe helper tests
│       └── utils.test.ts     # Utility function tests
├── components/
│   ├── ui/
│   │   └── __tests__/
│   │       ├── button.test.tsx
│   │       ├── card.test.tsx
│   │       ├── input.test.tsx
│   │       └── label.test.tsx
│   └── dashboard/
│       └── __tests__/
│           └── add-hallucination-form.test.tsx
└── app/
    └── __tests__/
        └── providers.test.tsx

e2e/
├── auth.spec.ts              # Authentication page tests
├── homepage.spec.ts          # Homepage tests
├── navigation.spec.ts        # Navigation and accessibility tests
├── pricing.spec.ts           # Pricing page tests
└── visual-and-interactions.spec.ts  # Visual regression & interaction tests
```

## Writing Tests

### Unit Test Example

```tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@/test/test-utils';
import { Button } from '@/components/ui/button';

describe('Button', () => {
  it('should render with text', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole('button', { name: /click me/i })).toBeInTheDocument();
  });
});
```

### E2E Test Example

```typescript
import { test, expect } from '@playwright/test';

test('should navigate to pricing page', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: /pricing/i }).click();
  await expect(page).toHaveURL(/.*pricing/);
});
```

### Mocking API Requests

```typescript
import { server } from '@/test/mocks/server';
import { http, HttpResponse } from 'msw';

it('should handle API error', async () => {
  server.use(
    http.get('/api/data', () => {
      return new HttpResponse('Error', { status: 500 });
    })
  );
  
  // Test error handling
});
```

## Coverage Thresholds

The project enforces the following coverage thresholds:

- **Lines**: 75%
- **Functions**: 70%
- **Branches**: 60%
- **Statements**: 75%

Coverage reports are generated in the `coverage/` directory.

## CI Integration

Add these scripts to your CI pipeline:

```yaml
# Run unit tests with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e

# Or run all tests
npm run test:all
```

## Best Practices

1. **Use the custom render function** from `@/test/test-utils` for component tests
2. **Mock external services** using MSW handlers
3. **Test user interactions** with `@testing-library/user-event`
4. **Write E2E tests** for critical user flows
5. **Keep tests isolated** - each test should be independent
6. **Use descriptive test names** that explain the expected behavior
7. **Test accessibility** - check for proper ARIA attributes and keyboard navigation

## Debugging Tests

### Vitest

```bash
# Run a specific test file
npx vitest src/lib/__tests__/api.test.ts

# Run tests matching a pattern
npx vitest -t "should verify text"
```

### Playwright

```bash
# Run a specific test file
npx playwright test e2e/auth.spec.ts

# Run in debug mode with Playwright Inspector
npx playwright test --debug

# Generate trace for debugging
npx playwright test --trace on
```
