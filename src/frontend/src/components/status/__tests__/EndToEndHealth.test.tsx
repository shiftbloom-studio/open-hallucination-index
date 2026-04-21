import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EndToEndHealth } from "../EndToEndHealth";
import { ohi } from "@/lib/ohi-client";
import { useHealthLive, useHealthReady } from "@/lib/ohi-queries";

vi.mock("@/lib/ohi-queries", () => ({
  useHealthLive: vi.fn(),
  useHealthReady: vi.fn(),
}));

vi.mock("@/lib/ohi-client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/ohi-client")>();
  return {
    ...actual,
    ohi: {
      ...actual.ohi,
      healthLive: vi.fn(),
      healthReady: vi.fn(),
      healthDeep: vi.fn(),
      verify: vi.fn(),
      verifyStatus: vi.fn(),
    },
  };
});

describe("EndToEndHealth", () => {
  const mockUseHealthLive = vi.mocked(useHealthLive);
  const mockUseHealthReady = vi.mocked(useHealthReady);

  beforeEach(() => {
    vi.clearAllMocks();

    mockUseHealthLive.mockReturnValue({
      data: { status: "healthy", timestamp: "2026-04-20T00:00:00Z" },
      error: null,
      isLoading: false,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useHealthLive>);

    mockUseHealthReady.mockReturnValue({
      data: { ready: true, timestamp: "2026-04-20T00:00:00Z", services: {} },
      error: null,
      isLoading: false,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useHealthReady>);

    vi.mocked(ohi.healthLive).mockResolvedValue({
      status: "healthy",
      timestamp: "2026-04-20T00:00:00Z",
    });
    vi.mocked(ohi.healthReady).mockResolvedValue({
      ready: true,
      timestamp: "2026-04-20T00:00:00Z",
      services: {
        llm: { connected: true, status: "healthy" },
      },
    });
    vi.mocked(ohi.healthDeep).mockResolvedValue({
      status: "ok",
      timestamp: "2026-04-20T00:00:00Z",
      layers: {
        "pipeline.orchestrator": { status: "ok", latency_ms: 10 },
      },
    });
    vi.mocked(ohi.verify).mockResolvedValue({ job_id: "job-12345678" });
    vi.mocked(ohi.verifyStatus).mockResolvedValue({
      job_id: "job-12345678",
      status: "done",
      phase: "assembling",
      created_at: Date.now(),
      updated_at: Date.now(),
      completed_at: Date.now(),
      result: {
        request_id: "r-1",
        pipeline_version: "test",
        model_versions: {},
        document_score: 0.9,
        document_interval: [0.8, 0.95],
        internal_consistency: 0.9,
        decomposition_coverage: 1,
        processing_time_ms: 321,
        rigor: "fast",
        refinement_passes_executed: 0,
        claims: [],
      },
    });
  });

  it("renders endpoint-level browser checks", () => {
    render(
      <EndToEndHealth
        deepData={{ status: "ok", timestamp: "2026-04-20T00:00:00Z", layers: {} }}
        deepLoading={false}
        onRefreshDeep={() => {}}
      />,
    );

    expect(screen.getByTestId("frontend-e2e-health")).toBeInTheDocument();
    expect(screen.getByText("Browser -> /health/live")).toBeInTheDocument();
    expect(screen.getByText("Browser -> /health/ready")).toBeInTheDocument();
    expect(screen.getByText("Browser -> /health/deep")).toBeInTheDocument();
  });

  it("runs the full frontend e2e probe flow", async () => {
    const onRefreshDeep = vi.fn();
    const user = userEvent.setup();

    render(
      <EndToEndHealth
        deepData={{ status: "ok", timestamp: "2026-04-20T00:00:00Z", layers: {} }}
        deepLoading={false}
        onRefreshDeep={onRefreshDeep}
      />,
    );

    await user.click(screen.getByTestId("run-e2e-probe"));

    await waitFor(() => {
      expect(screen.getByTestId("e2e-probe-status")).toHaveTextContent("probe ok");
    });

    expect(ohi.healthLive).toHaveBeenCalled();
    expect(ohi.healthReady).toHaveBeenCalled();
    expect(ohi.healthDeep).toHaveBeenCalled();
    expect(ohi.verify).toHaveBeenCalled();
    expect(ohi.verifyStatus).toHaveBeenCalled();
    expect(onRefreshDeep).toHaveBeenCalled();
  });
});
