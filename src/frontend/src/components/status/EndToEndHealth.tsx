"use client";

import { useMemo, useState } from "react";
import { OhiError, ohi } from "@/lib/ohi-client";
import { useHealthLive, useHealthReady } from "@/lib/ohi-queries";
import type { HealthDeep } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";

type StepState = "idle" | "checking" | "ok" | "degraded" | "down";

interface ProbeStep {
  id: string;
  label: string;
  state: StepState;
  detail?: string;
  durationMs?: number;
}

interface EndpointRow {
  label: string;
  state: StepState;
  detail: string;
}

const BASE_STEPS: ProbeStep[] = [
  { id: "live", label: "GET /health/live", state: "idle" },
  { id: "ready", label: "GET /health/ready", state: "idle" },
  { id: "deep", label: "GET /health/deep", state: "idle" },
  { id: "verify", label: "POST /api/v2/verify + poll status", state: "idle" },
];

const VERIFY_POLL_INTERVAL_MS = 2000;
const VERIFY_PROBE_TIMEOUT_MS = 90_000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatErrorDetail(error: unknown): string {
  if (error instanceof OhiError) {
    const bodyDetail =
      typeof (error.body as { detail?: unknown }).detail === "string"
        ? (error.body as { detail: string }).detail
        : null;
    if (error.isResting) {
      const retry = error.retryAfterSec ? `retry-after ${error.retryAfterSec}s` : "retry-after n/a";
      return `resting (${retry})`;
    }
    if (error.isRateLimited) {
      return "rate limited (429)";
    }
    if (bodyDetail) {
      return `HTTP ${error.status}: ${bodyDetail}`;
    }
    return `HTTP ${error.status}`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function stateBadge(state: StepState): string {
  if (state === "ok")
    return "border-[color:var(--brand-success)]/30 bg-[color:var(--brand-success-soft)] text-[color:var(--brand-success)]";
  if (state === "degraded")
    return "border-[color:var(--brand-warning)]/30 bg-[color:var(--brand-warning-soft)] text-[color:var(--brand-warning)]";
  if (state === "down")
    return "border-[color:var(--brand-danger)]/30 bg-[color:var(--brand-danger-soft)] text-[color:var(--brand-danger)]";
  if (state === "checking")
    return "border-[color:var(--brand-info)]/25 bg-[color:var(--brand-info-soft)] text-[color:var(--brand-info)]";
  return "border-[color:var(--border-default)] bg-[color:var(--surface-soft)] text-[color:var(--brand-muted)]";
}

function stateLabel(state: StepState): string {
  if (state === "ok") return "ok";
  if (state === "degraded") return "degraded";
  if (state === "down") return "down";
  if (state === "checking") return "checking";
  return "idle";
}

function shortJobId(id: string): string {
  return id.slice(0, 8);
}

export interface EndToEndHealthProps {
  deepData?: HealthDeep;
  deepError?: unknown;
  deepLoading: boolean;
  onRefreshDeep: () => void;
  className?: string;
}

export function EndToEndHealth({
  deepData,
  deepError,
  deepLoading,
  onRefreshDeep,
  className,
}: EndToEndHealthProps) {
  const liveQuery = useHealthLive();
  const readyQuery = useHealthReady();
  const [runningProbe, setRunningProbe] = useState(false);
  const [steps, setSteps] = useState<ProbeStep[]>(BASE_STEPS);
  const [lastProbeAt, setLastProbeAt] = useState<string | null>(null);

  const updateStep = (id: string, patch: Partial<ProbeStep>) => {
    setSteps((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  };

  const endpointRows = useMemo<EndpointRow[]>(
    () => [
      {
        label: "Browser -> /health/live",
        state: liveQuery.isLoading
          ? "checking"
          : liveQuery.error
            ? "down"
            : liveQuery.data?.status === "healthy"
              ? "ok"
              : "degraded",
        detail: liveQuery.error
          ? formatErrorDetail(liveQuery.error)
          : liveQuery.data
            ? `status=${liveQuery.data.status}`
            : "waiting",
      },
      {
        label: "Browser -> /health/ready",
        state: readyQuery.isLoading
          ? "checking"
          : readyQuery.error
            ? "down"
            : readyQuery.data?.ready
              ? "ok"
              : "degraded",
        detail: readyQuery.error
          ? formatErrorDetail(readyQuery.error)
          : readyQuery.data
            ? `ready=${String(readyQuery.data.ready)}`
            : "waiting",
      },
      {
        label: "Browser -> /health/deep",
        state: deepLoading
          ? "checking"
          : deepError
            ? "down"
            : deepData
              ? deepData.status === "ok"
                ? "ok"
                : deepData.status === "degraded"
                  ? "degraded"
                  : "down"
              : "idle",
        detail: deepError
          ? formatErrorDetail(deepError)
          : deepData
            ? `status=${deepData.status ?? deepData.overall ?? "unknown"}`
            : "waiting",
      },
    ],
    [deepData, deepError, deepLoading, liveQuery.data, liveQuery.error, liveQuery.isLoading, readyQuery.data, readyQuery.error, readyQuery.isLoading],
  );

  const probeOverall: StepState = useMemo(() => {
    if (runningProbe) return "checking";
    if (steps.some((s) => s.state === "down")) return "down";
    if (steps.some((s) => s.state === "degraded")) return "degraded";
    if (steps.every((s) => s.state === "ok")) return "ok";
    if (steps.some((s) => s.state === "checking")) return "checking";
    return "idle";
  }, [runningProbe, steps]);

  const runProbe = async () => {
    if (runningProbe) return;
    setRunningProbe(true);
    setLastProbeAt(new Date().toISOString());
    setSteps(BASE_STEPS.map((s) => ({ ...s })));

    const runStep = async (
      id: string,
      work: () => Promise<{ state: Exclude<StepState, "idle" | "checking">; detail: string }>,
    ) => {
      updateStep(id, { state: "checking", detail: undefined, durationMs: undefined });
      const started = performance.now();
      try {
        const { state, detail } = await work();
        updateStep(id, {
          state,
          detail,
          durationMs: Math.round(performance.now() - started),
        });
        return state;
      } catch (error) {
        updateStep(id, {
          state: "down",
          detail: formatErrorDetail(error),
          durationMs: Math.round(performance.now() - started),
        });
        return "down";
      }
    };

    await runStep("live", async () => {
      const live = await ohi.healthLive();
      if (live.status === "healthy") {
        return { state: "ok", detail: `healthy @ ${live.timestamp}` };
      }
      if (live.status === "degraded") {
        return { state: "degraded", detail: `degraded @ ${live.timestamp}` };
      }
      throw new Error(`unexpected status ${live.status}`);
    });

    await runStep("ready", async () => {
      const ready = await ohi.healthReady();
      const connected = Object.entries(ready.services)
        .filter(([, svc]) => svc.connected)
        .length;
      if (!ready.ready) {
        return { state: "degraded", detail: `ready=false (${connected} services connected)` };
      }
      return { state: "ok", detail: `ready=true (${connected} services connected)` };
    });

    await runStep("deep", async () => {
      const deep = await ohi.healthDeep();
      const status = deep.status ?? deep.overall;
      if (status === "ok" || status === "healthy") {
        return { state: "ok", detail: `ok with ${Object.keys(deep.layers).length} layers` };
      }
      if (status === "degraded") {
        return {
          state: "degraded",
          detail: `degraded with ${Object.keys(deep.layers).length} layers`,
        };
      }
      if (status !== "down" && status !== "unhealthy") {
        throw new Error(`unexpected status ${deep.status ?? deep.overall ?? "unknown"}`);
      }
      throw new Error(`status=${status}`);
    });

    await runStep("verify", async () => {
      const accepted = await ohi.verify({
        text: "The Pacific Ocean is the largest ocean on Earth.",
        options: {
          rigor: "fast",
          max_claims: 1,
          include_pcg_neighbors: false,
          include_full_provenance: false,
        },
      });

      const deadline = Date.now() + VERIFY_PROBE_TIMEOUT_MS;
      while (Date.now() < deadline) {
        const status = await ohi.verifyStatus(accepted.job_id);
        if (status.status === "done") {
          const ms = status.result?.processing_time_ms;
          return {
            state: "ok",
            detail: ms
              ? `job ${shortJobId(status.job_id)} done (${ms} ms)`
              : `job ${shortJobId(status.job_id)} done`,
          };
        }
        if (status.status === "error") {
          throw new Error(status.error ?? "verify job failed");
        }
        await sleep(VERIFY_POLL_INTERVAL_MS);
      }

      throw new Error(`timeout after ${VERIFY_PROBE_TIMEOUT_MS / 1000}s`);
    });

    setRunningProbe(false);
    void liveQuery.refetch();
    void readyQuery.refetch();
    onRefreshDeep();
  };

  return (
    <section
      className={cn(
        "mb-6 overflow-hidden rounded-xl border border-[color:var(--border-subtle)] bg-[color:var(--surface-elevated)] p-5 shadow-sm",
        className,
      )}
      data-testid="frontend-e2e-health"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-[color:var(--brand-ink)]">
            Frontend end-to-end health
          </h2>
          <p className="mt-1 text-xs text-[color:var(--brand-muted)]">
            Browser checks for live/ready/deep plus an optional real verify smoke run.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "rounded-full border px-2 py-0.5 text-[10px] font-mono uppercase tracking-wide",
              stateBadge(probeOverall),
            )}
            data-testid="e2e-probe-status"
          >
            probe {stateLabel(probeOverall)}
          </span>
          <button
            type="button"
            onClick={runProbe}
            disabled={runningProbe}
            className="rounded-md border border-[color:var(--border-default)] bg-[color:var(--surface-soft)] px-3 py-1.5 text-xs font-semibold text-[color:var(--brand-ink)] transition hover:border-[color:var(--border-accent)] hover:bg-[color:var(--brand-indigo-soft)] disabled:cursor-not-allowed disabled:opacity-50"
            data-testid="run-e2e-probe"
          >
            {runningProbe ? "Running probe..." : "Run end-to-end probe"}
          </button>
        </div>
      </div>

      <ol className="mt-4 grid gap-2 text-xs text-[color:var(--brand-ink)]">
        {endpointRows.map((row) => (
          <li
            key={row.label}
            className="flex flex-wrap items-center gap-2 rounded-md border border-[color:var(--border-subtle)] bg-[color:var(--surface-soft)]/40 px-3 py-2"
          >
            <span className="font-mono text-[color:var(--brand-ink)]">{row.label}</span>
            <span className={cn("rounded-full border px-2 py-0.5 text-[10px] font-mono uppercase", stateBadge(row.state))}>
              {stateLabel(row.state)}
            </span>
            <span className="ml-auto font-mono text-[10px] text-[color:var(--brand-muted)]">
              {row.detail}
            </span>
          </li>
        ))}
      </ol>

      <ol className="mt-4 grid gap-2 text-xs text-[color:var(--brand-ink)]">
        {steps.map((step) => (
          <li
            key={step.id}
            className="flex flex-wrap items-center gap-2 rounded-md border border-[color:var(--border-subtle)] bg-[color:var(--surface-soft)]/40 px-3 py-2"
            data-testid={`e2e-probe-step-${step.id}`}
          >
            <span className="font-mono text-[color:var(--brand-ink)]">{step.label}</span>
            <span className={cn("rounded-full border px-2 py-0.5 text-[10px] font-mono uppercase", stateBadge(step.state))}>
              {stateLabel(step.state)}
            </span>
            {typeof step.durationMs === "number" && (
              <span className="font-mono text-[10px] text-[color:var(--brand-subtle)]">
                {step.durationMs} ms
              </span>
            )}
            {step.detail && (
              <span
                className="ml-auto max-w-[48ch] truncate font-mono text-[10px] text-[color:var(--brand-muted)]"
                title={step.detail}
              >
                {step.detail}
              </span>
            )}
          </li>
        ))}
      </ol>

      <p className="mt-3 text-[10px] text-[color:var(--brand-subtle)]">
        {lastProbeAt ? `Last probe started at ${lastProbeAt}. ` : ""}
        The verify probe triggers one real backend job and can consume model quota.
      </p>
    </section>
  );
}
