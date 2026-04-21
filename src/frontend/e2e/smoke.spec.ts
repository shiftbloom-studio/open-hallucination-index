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

test("status page renders frontend end-to-end health controls", async ({ page }) => {
  await page.goto("/status");
  await expect(page.getByTestId("frontend-e2e-health")).toBeVisible();
  await expect(page.getByTestId("run-e2e-probe")).toBeVisible();
});

test.describe("live backend probe", () => {
  test.skip(process.env.RUN_LIVE_E2E_STATUS !== "1", "set RUN_LIVE_E2E_STATUS=1 for real backend probe");

  test("status page probe reaches backend end-to-end", async ({ page }) => {
    await page.goto("/status");
    await page.getByTestId("run-e2e-probe").click();
    await expect(page.getByTestId("e2e-probe-status")).toHaveText(/probe ok/i, {
      timeout: 120_000,
    });
  });
});
