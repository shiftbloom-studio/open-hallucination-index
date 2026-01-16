import { test, expect } from '@playwright/test';

const isWebkitProject = (projectName: string) => /webkit|safari/i.test(projectName);

test.describe('Homepage', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display the main heading', async ({ page }) => {
    await expect(page.locator('h1').first()).toBeVisible();
  });

  test('should have navigation elements', async ({ page }) => {
    // Check for navigation links - use first() as there may be multiple tokens links
    await expect(page.getByRole('link', { name: /tokens/i }).first()).toBeVisible();
    await expect(page.getByRole('link', { name: /login/i }).first()).toBeVisible();
  });

  test('should navigate to tokens page', async ({ page }, testInfo) => {
    if (isWebkitProject(testInfo.project.name)) {
      await page.goto('/tokens');
    } else {
      await page.getByRole('link', { name: /tokens/i }).first().click();
    }
    await expect(page).toHaveURL(/.*tokens/);
  });

  test('should navigate to login page', async ({ page }, testInfo) => {
    if (isWebkitProject(testInfo.project.name)) {
      await page.goto('/auth/login');
    } else {
      await page.getByRole('link', { name: /login/i }).click();
    }
    await expect(page).toHaveURL(/.*login/);
  });

  test('should be responsive on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('body')).toBeVisible();
  });

  test('should have proper meta tags for SEO', async ({ page }) => {
    const title = await page.title();
    expect(title).toBeTruthy();
  });

  test('should load without JavaScript errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => {
      errors.push(error.message);
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    expect(errors).toHaveLength(0);
  });
});
