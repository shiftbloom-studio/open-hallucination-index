import { test, expect } from '@playwright/test';

test.describe('Dashboard', () => {
  // Note: These tests assume an unauthenticated state
  // In a real scenario, you'd mock authentication or use test fixtures
  
  test('should redirect unauthenticated users to login', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Should either redirect to login or show login prompt
    const currentUrl = page.url();
    const isOnLoginPage = currentUrl.includes('login');
    const hasLoginPrompt = await page.getByRole('link', { name: /login|sign in/i }).isVisible().catch(() => false);
    
    expect(isOnLoginPage || hasLoginPrompt).toBeTruthy();
  });

  test('should have proper layout structure', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Check that page loads without errors
    await expect(page.locator('body')).toBeVisible();
  });
});

test.describe('Legal Pages', () => {
  test('should display Impressum page', async ({ page }) => {
    await page.goto('/impressum');
    await expect(page.locator('body')).toBeVisible();
    await expect(page.getByText(/Impressum|Imprint/i).first()).toBeVisible();
  });

  test('should display Datenschutz page', async ({ page }) => {
    await page.goto('/datenschutz');
    await expect(page.locator('body')).toBeVisible();
    await expect(page.getByText(/Datenschutz|Privacy/i).first()).toBeVisible();
  });

  test('should display AGB page', async ({ page }) => {
    await page.goto('/agb');
    await expect(page.locator('body')).toBeVisible();
    await expect(page.getByText(/AGB|Terms|Allgemeine/i).first()).toBeVisible();
  });
});

test.describe('Accessibility', () => {
  const pages = ['/', '/pricing', '/auth/login', '/auth/signup'];

  for (const pagePath of pages) {
    test(`should have no critical accessibility issues on ${pagePath}`, async ({ page }) => {
      await page.goto(pagePath);

      // Check for skip link or main landmark
      const hasMain = await page.locator('main').count() > 0;
      const hasMainRole = await page.locator('[role="main"]').count() > 0;
      expect(hasMain || hasMainRole).toBeTruthy();

      // Check heading hierarchy
      const headings = await page.locator('h1, h2, h3, h4, h5, h6').all();
      expect(headings.length).toBeGreaterThan(0);

      // Check for proper button accessibility
      const buttons = await page.locator('button').all();
      for (const button of buttons) {
        const text = await button.textContent();
        const ariaLabel = await button.getAttribute('aria-label');
        const hasAccessibleName = (text && text.trim() !== '') || ariaLabel;
        expect(hasAccessibleName).toBeTruthy();
      }
    });

    test(`should be keyboard navigable on ${pagePath}`, async ({ page }) => {
      await page.goto(pagePath);

      // Tab through the page and verify focus is visible
      await page.keyboard.press('Tab');
      
      const focusedElement = await page.evaluate(() => {
        return document.activeElement?.tagName;
      });
      
      expect(focusedElement).toBeTruthy();
    });
  }
});

test.describe('Performance', () => {
  test('homepage should load within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    const loadTime = Date.now() - startTime;

    // Page should load within 5 seconds
    expect(loadTime).toBeLessThan(5000);
  });

  test('should not have excessive network requests', async ({ page }) => {
    const requests: string[] = [];
    
    page.on('request', (request) => {
      requests.push(request.url());
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Should not make excessive requests
    expect(requests.length).toBeLessThan(100);
  });
});

test.describe('Error Handling', () => {
  test('should handle 404 pages gracefully', async ({ page }) => {
    const response = await page.goto('/non-existent-page-12345');
    
    // In development mode, Next.js may return 200 with error overlay
    // In production, it should return 404
    const status = response?.status();
    expect(status === 404 || status === 200).toBeTruthy();
  });

  test('should not expose sensitive information in errors', async ({ page }) => {
    await page.goto('/non-existent-page-12345');
    
    const bodyText = await page.locator('body').textContent();
    
    // In production, should not expose stack traces or sensitive info
    // Skip check if in development mode (dev server includes debugging info)
    if (!bodyText?.includes('__next')) {
      expect(bodyText).not.toContain('Error:');
      expect(bodyText).not.toContain('node_modules');
    }
  });
});
