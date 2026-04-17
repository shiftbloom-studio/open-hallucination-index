import { test, expect } from "@playwright/test";

// Minimal smoke tests. Full coverage (streaming /verify, resting state,
// calibration, status) is deferred to a post-touch-up follow-up per
// docs/superpowers/specs/2026-04-17-ohi-v2-frontend-design.md §9.

test("landing page loads and renders hero copy", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText(/How much should you/i)).toBeVisible();
});

test("navbar links to /verify, /calibration, /status", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("link", { name: /^Verify$/i })).toBeVisible();
  await expect(page.getByRole("link", { name: /^Calibration$/i })).toBeVisible();
  await expect(page.getByRole("link", { name: /^Status$/i })).toBeVisible();
});
