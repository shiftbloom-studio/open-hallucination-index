import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { IntervalBar } from "../IntervalBar";
import { DomainBadge } from "../DomainBadge";
import { FallbackBadge } from "../FallbackBadge";

describe("IntervalBar", () => {
  it("positions the fill using the interval range", () => {
    render(<IntervalBar pTrue={0.7} interval={[0.4, 0.9]} />);
    const fill = screen.getByTestId("interval-fill") as HTMLDivElement;
    expect(fill.style.left).toBe("40%");
    expect(fill.style.width).toBe("50%");
  });

  it("uses the high-band class when p_true ≥ 0.8", () => {
    const { container } = render(<IntervalBar pTrue={0.96} interval={[0.91, 0.99]} />);
    const bar = container.firstChild as HTMLElement;
    expect(bar.getAttribute("data-band")).toBe("high");
  });

  it("uses mid-band for 0.5 ≤ p < 0.8", () => {
    const { container } = render(<IntervalBar pTrue={0.62} interval={[0.38, 0.81]} />);
    expect((container.firstChild as HTMLElement).getAttribute("data-band")).toBe("mid");
  });

  it("uses low-band for p < 0.5", () => {
    const { container } = render(<IntervalBar pTrue={0.2} interval={[0.05, 0.4]} />);
    expect((container.firstChild as HTMLElement).getAttribute("data-band")).toBe("low");
  });

  it('applies compact height when size="sm"', () => {
    const { container } = render(<IntervalBar pTrue={0.7} interval={[0.5, 0.9]} size="sm" />);
    expect((container.firstChild as HTMLElement).className).toContain("h-1");
  });

  it("clamps out-of-range values", () => {
    render(<IntervalBar pTrue={1.3} interval={[-0.1, 1.5]} />);
    const fill = screen.getByTestId("interval-fill") as HTMLDivElement;
    expect(fill.style.left).toBe("0%");
    expect(fill.style.width).toBe("100%");
  });
});

describe("DomainBadge", () => {
  it("renders the domain name", () => {
    render(<DomainBadge domain="biomedical" />);
    expect(screen.getByText("biomedical")).toBeInTheDocument();
  });

  it("shows weight when < 1", () => {
    render(<DomainBadge domain="general" weight={0.85} />);
    expect(screen.getByText(/·0\.85/)).toBeInTheDocument();
  });

  it("hides weight when exactly 1", () => {
    render(<DomainBadge domain="general" weight={1} />);
    expect(screen.queryByText(/·1\.00/)).not.toBeInTheDocument();
  });
});

describe("FallbackBadge", () => {
  it("renders nothing when kind is null", () => {
    const { container } = render(<FallbackBadge kind={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("shows the fallback label with a warning glyph", () => {
    render(<FallbackBadge kind="general" />);
    expect(screen.getByText(/general fallback/)).toBeInTheDocument();
  });
});
