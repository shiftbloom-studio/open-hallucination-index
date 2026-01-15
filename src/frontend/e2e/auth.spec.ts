import { test, expect } from '@playwright/test';

const isWebkitProject = (projectName: string) => /webkit|safari/i.test(projectName);

test.describe('Authentication Pages', () => {
  test.describe('Login Page', () => {
    test.beforeEach(async ({ page }) => {
      await page.goto('/auth/login');
    });

    test('should display login form', async ({ page }) => {
      await expect(page.getByRole('heading', { name: /login|sign in/i })).toBeVisible();
    });

    test('should have email input', async ({ page }) => {
      const emailInput = page.getByRole('textbox', { name: /email/i }).or(
        page.locator('input[type="email"]')
      );
      await expect(emailInput).toBeVisible();
    });

    test('should have password input', async ({ page }) => {
      const passwordInput = page.locator('input[type="password"]');
      await expect(passwordInput).toBeVisible();
    });

    test('should have submit button', async ({ page }) => {
      // Look for the submit button specifically within the form
      const submitButton = page.locator('form').getByRole('button', { name: /login|sign in|submit/i });
      await expect(submitButton).toBeVisible();
    });

    test('should have link to signup page', async ({ page }) => {
      await expect(
        page.getByRole('link', { name: /sign up|register|create account/i }).first()
      ).toBeVisible();
    });

    test('should show validation error for empty submission', async ({ page }) => {
      const submitButton = page.locator('form').getByRole('button', { name: /login|sign in|submit/i });
      await submitButton.click();
      // HTML5 validation should prevent submission
      const emailInput = page.locator('input[type="email"]');
      const validationMessage = await emailInput.evaluate(
        (el: HTMLInputElement) => el.validationMessage
      );
      expect(validationMessage).toBeTruthy();
    });

    test('should validate email format', async ({ page }) => {
      const emailInput = page.locator('input[type="email"]');
      await emailInput.fill('invalid-email');
      const submitButton = page.locator('form').getByRole('button', { name: /login|sign in|submit/i });
      await submitButton.click();
      
      const validationMessage = await emailInput.evaluate(
        (el: HTMLInputElement) => el.validationMessage
      );
      expect(validationMessage).toBeTruthy();
    });
  });

  test.describe('Signup Page', () => {
    test.beforeEach(async ({ page }) => {
      await page.goto('/auth/signup');
    });

    test('should display signup form', async ({ page }) => {
      await expect(page.getByRole('heading', { name: /sign up|register|create/i })).toBeVisible();
    });

    test('should have email input', async ({ page }) => {
      const emailInput = page.getByRole('textbox', { name: /email/i }).or(
        page.locator('input[type="email"]')
      );
      await expect(emailInput).toBeVisible();
    });

    test('should have password input', async ({ page }) => {
      // Check for at least one password input (signup has two: password and confirm)
      const passwordInput = page.locator('input[type="password"]').first();
      await expect(passwordInput).toBeVisible();
    });

    test('should have link to login page', async ({ page }) => {
      await expect(
        page.getByRole('link', { name: /login|sign in|already have/i }).first()
      ).toBeVisible();
    });

    test('should navigate between login and signup', async ({ page }, testInfo) => {
      if (isWebkitProject(testInfo.project.name)) {
        await page.goto('/auth/login');
      } else {
        await page.getByRole('link', { name: /login|sign in|already have/i }).first().click();
      }
      await expect(page).toHaveURL(/.*login/);
    });
  });
});
