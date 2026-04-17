import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";

// next/link tries to observe intersection for prefetch; the global mock in
// test/setup.ts gives an arrow that can't be called with `new` on some paths.
// Stub Link to a plain <a> for these purely-visual components.
vi.mock("next/link", () => ({
  __esModule: true,
  default: ({ children, href, ...rest }: { children: React.ReactNode; href: string }) => (
    <a href={typeof href === "string" ? href : "#"} {...rest}>
      {children}
    </a>
  ),
}));

import { RestingState } from "../RestingState";
import { BudgetExhaustedState } from "../BudgetExhaustedState";
import { LlmUnavailableState } from "../LlmUnavailableState";
import { RateLimitedState } from "../RateLimitedState";
import { NetworkErrorState } from "../NetworkErrorState";
import { DegradedState } from "../DegradedState";
import { RetryCountdown } from "../RetryCountdown";

describe("error-state components", () => {
  it("RestingState: renders and hosts retry countdown", () => {
    render(<RestingState retryAfterSec={60} />);
    expect(screen.getByTestId("resting-state")).toBeInTheDocument();
    expect(screen.getByText(/resting/i)).toBeInTheDocument();
  });

  it("BudgetExhaustedState renders", () => {
    render(<BudgetExhaustedState />);
    expect(screen.getByTestId("budget-exhausted-state")).toBeInTheDocument();
  });

  it("LlmUnavailableState renders", () => {
    render(<LlmUnavailableState />);
    expect(screen.getByTestId("llm-unavailable-state")).toBeInTheDocument();
  });

  it("RateLimitedState renders", () => {
    render(<RateLimitedState retryAfterSec={30} />);
    expect(screen.getByTestId("rate-limited-state")).toBeInTheDocument();
  });

  it("NetworkErrorState renders and fires retry callback", () => {
    const onRetry = vi.fn();
    render(<NetworkErrorState onRetrySync={onRetry} />);
    fireEvent.click(screen.getByRole("button", { name: /retry with sync/i }));
    expect(onRetry).toHaveBeenCalled();
  });

  it("DegradedState renders nothing when there's nothing to warn about", () => {
    const { container } = render(<DegradedState degradedLayers={[]} fallbackCount={0} />);
    expect(container.firstChild).toBeNull();
  });

  it("DegradedState lists degraded layers", () => {
    render(<DegradedState degradedLayers={["L3", "L1.retrieval"]} fallbackCount={2} />);
    expect(screen.getByText("L3")).toBeInTheDocument();
    expect(screen.getByText("L1.retrieval")).toBeInTheDocument();
    expect(screen.getByText(/2 claim/i)).toBeInTheDocument();
  });
});

describe("RetryCountdown", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("disables retry until countdown elapses", () => {
    const onRetry = vi.fn();
    render(<RetryCountdown retryAfterSec={3} onRetry={onRetry} />);
    const btn = screen.getByRole("button");
    expect(btn).toBeDisabled();

    act(() => {
      vi.advanceTimersByTime(3000);
    });
    expect(btn).not.toBeDisabled();
    fireEvent.click(btn);
    expect(onRetry).toHaveBeenCalled();
  });

  it("renders an enabled retry button when retryAfterSec is 0/undefined", () => {
    const onRetry = vi.fn();
    render(<RetryCountdown onRetry={onRetry} />);
    expect(screen.getByRole("button")).not.toBeDisabled();
  });
});
