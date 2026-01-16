import { test, expect } from '@playwright/test';

test.describe('Pricing Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/pricing');
  });

  test('should display tokens heading', async ({ page }) => {
    await expect(page.locator('h1, h2').first()).toBeVisible();
  });

  test('should display all three tokens packages', async ({ page }) => {
    // Check for the three token packages
    await expect(page.getByText('10').first()).toBeVisible();
    await expect(page.getByText('100').first()).toBeVisible();
    await expect(page.getByText('500').first()).toBeVisible();
  });

  test('should display prices for each package', async ({ page }) => {
    await expect(page.getByText(/1.49€|1,49€/)).toBeVisible();
    await expect(page.getByText(/9.99€|9,99€/)).toBeVisible();
    await expect(page.getByText(/24.99€|24,99€/)).toBeVisible();
  });

  test('should have purchase buttons', async ({ page }) => {
    const buttons = page.locator('button');
    const count = await buttons.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should display FAQ section', async ({ page }) => {
    await expect(page.getByText(/FAQ|Frequently Asked/i)).toBeVisible();
  });

  test('should expand FAQ items on click', async ({ page }) => {
    const faqQuestion = page.getByText('What is an OHI Token?');
    if (await faqQuestion.isVisible()) {
      await faqQuestion.click();
      // Check if answer is visible after click
      await expect(page.getByText(/verification request/i).first()).toBeVisible();
    }
  });

  test('should show best value tag on premium package', async ({ page }) => {
    await expect(page.getByText(/Best Value|Save/i).first()).toBeVisible();
  });

  test('should display trust badges', async ({ page }) => {
    await expect(page.getByText(/SSL|GDPR|Fair Terms/i).first()).toBeVisible();
  });

  test('should be accessible', async ({ page }) => {
    // Check for proper heading structure
    const h1Count = await page.locator('h1').count();
    const h2Count = await page.locator('h2').count();
    expect(h1Count + h2Count).toBeGreaterThan(0);

    // Check for alt text on images
    const images = page.locator('img');
    const imgCount = await images.count();
    for (let i = 0; i < imgCount; i++) {
      const alt = await images.nth(i).getAttribute('alt');
      // Images should have alt attribute (can be empty for decorative images)
      expect(alt).not.toBeNull();
    }
  });
});
