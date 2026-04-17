import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { VerifyForm } from "../VerifyForm";

describe("VerifyForm", () => {
  it("submit disabled until text is entered", () => {
    const onSubmit = vi.fn();
    render(<VerifyForm onSubmit={onSubmit} />);
    expect(screen.getByRole("button", { name: /verify/i })).toBeDisabled();
  });

  it("emits VerifyRequest with selected rigor, domain_hint, coverage_target", async () => {
    const onSubmit = vi.fn();
    render(<VerifyForm onSubmit={onSubmit} />);
    await userEvent.type(screen.getByRole("textbox"), "hello world");
    // default rigor = balanced; switch to fast
    await userEvent.click(screen.getByRole("button", { name: /^fast$/i }));
    await userEvent.click(screen.getByRole("button", { name: /^biomedical$/i }));
    await userEvent.click(screen.getByRole("button", { name: /^95%$/i }));
    await userEvent.click(screen.getByRole("button", { name: /verify/i }));
    expect(onSubmit).toHaveBeenCalledWith({
      text: "hello world",
      domain_hint: "biomedical",
      options: { rigor: "fast", coverage_target: 0.95 },
    });
  });

  it("shows 'Cancel' and calls onCancel while streaming", () => {
    const onCancel = vi.fn();
    render(<VerifyForm onSubmit={() => {}} onCancel={onCancel} streaming />);
    const cancel = screen.getByRole("button", { name: /cancel/i });
    fireEvent.click(cancel);
    expect(onCancel).toHaveBeenCalled();
  });

  it("char counter turns rose + submit disabled over 50k chars", () => {
    render(<VerifyForm onSubmit={() => {}} />);
    const ta = screen.getByRole("textbox");
    fireEvent.change(ta, { target: { value: "a".repeat(50_001) } });
    expect(screen.getByText(/Too long/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /verify/i })).toBeDisabled();
  });
});
