import { describe, it, expect, beforeAll, afterAll, afterEach, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";
import { FeedbackSheet } from "../FeedbackSheet";
import type { ClaimVerdict, DocumentVerdict } from "@/lib/ohi-types";
import golden from "@/test/fixtures/document-verdict.golden.json";

const API_BASE = "https://api.ohi.test/api/v2";
const server = setupServer();

beforeAll(() => {
  process.env.NEXT_PUBLIC_API_BASE = API_BASE;
  server.listen({ onUnhandledRequest: "error" });
});
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

const doc = golden as unknown as DocumentVerdict;
const [c1] = doc.claims as [ClaimVerdict];

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>;
}

describe("FeedbackSheet", () => {
  it("renders nothing when open is false", () => {
    render(
      wrap(
        <FeedbackSheet requestId={doc.request_id} claim={c1} open={false} onClose={() => {}} />,
      ),
    );
    expect(screen.queryByTestId("feedback-sheet")).not.toBeInTheDocument();
  });

  it("renders label options and claim text when open", () => {
    render(
      wrap(
        <FeedbackSheet requestId={doc.request_id} claim={c1} open={true} onClose={() => {}} />,
      ),
    );
    expect(screen.getByTestId("feedback-sheet")).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /true/i })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /false/i })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /unverifiable/i })).toBeInTheDocument();
  });

  it("shows char counter; disables submit when rationale > 2000 chars", async () => {
    render(
      wrap(
        <FeedbackSheet requestId={doc.request_id} claim={c1} open={true} onClose={() => {}} />,
      ),
    );
    const ta = screen.getByRole("textbox");
    // maxLength on the element itself caps at 2500; we test the over-2000 styling
    const over = "a".repeat(2001);
    fireEvent.change(ta, { target: { value: over } });
    expect(screen.getByText(/2001 \/ 2000/)).toBeInTheDocument();
    const submit = screen.getByRole("button", { name: /submit/i });
    expect(submit).toBeDisabled();
  });

  it("POSTs /feedback with (request_id, claim_id, labeler.id) on submit", async () => {
    const posted: unknown[] = [];
    server.use(
      http.post(`${API_BASE}/feedback`, async ({ request }) => {
        posted.push(await request.json());
        return HttpResponse.json({ feedback_id: "fb1", queued: true }, { status: 202 });
      }),
    );

    // Seed labeler id for determinism (global setup mocks localStorage as vi.fn())
    vi.mocked(window.localStorage.getItem).mockReturnValue("labeler-abc");

    render(
      wrap(
        <FeedbackSheet requestId={doc.request_id} claim={c1} open={true} onClose={() => {}} />,
      ),
    );

    await userEvent.click(screen.getByRole("radio", { name: /false/i }));
    await userEvent.click(screen.getByRole("button", { name: /submit/i }));

    await waitFor(() => {
      expect(posted).toHaveLength(1);
    });
    const body = posted[0] as Record<string, unknown>;
    expect(body.request_id).toBe(doc.request_id);
    expect(body.claim_id).toBe(c1.claim.id);
    expect(body.label).toBe("false");
    expect((body.labeler as { id: string }).id).toBe("labeler-abc");
  });

  it("calls onClose on Escape", () => {
    const onClose = vi.fn();
    render(
      wrap(<FeedbackSheet requestId={doc.request_id} claim={c1} open={true} onClose={onClose} />),
    );
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });
});
