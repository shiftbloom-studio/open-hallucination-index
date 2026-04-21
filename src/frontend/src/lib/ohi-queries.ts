import { useMutation, useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { ohi } from "./ohi-client";
import type {
  CalibrationReport,
  FeedbackRequest,
  HealthDeep,
  HealthStatus,
  ReadinessStatus,
} from "./ohi-types";

export const queryKeys = {
  calibration: ["calibration", "report"] as const,
  health: ["health", "deep"] as const,
  healthLive: ["health", "live"] as const,
  healthReady: ["health", "ready"] as const,
  verdict: (id: string) => ["verdict", id] as const,
};

export function useCalibration(
  options?: Omit<UseQueryOptions<CalibrationReport>, "queryKey" | "queryFn">,
) {
  return useQuery<CalibrationReport>({
    queryKey: queryKeys.calibration,
    queryFn: ({ signal }) => ohi.calibrationReport({ signal }),
    staleTime: 5 * 60 * 1000, // 5 min
    gcTime: 10 * 60 * 1000,
    ...options,
  });
}

export function useHealthDeep(
  options?: Omit<UseQueryOptions<HealthDeep>, "queryKey" | "queryFn">,
) {
  return useQuery<HealthDeep>({
    queryKey: queryKeys.health,
    queryFn: ({ signal }) => ohi.healthDeep({ signal }),
    staleTime: 30 * 1000, // 30s — cheap probe
    refetchInterval: 30 * 1000,
    retry: false,
    ...options,
  });
}

export function useHealthLive(
  options?: Omit<UseQueryOptions<HealthStatus>, "queryKey" | "queryFn">,
) {
  return useQuery<HealthStatus>({
    queryKey: queryKeys.healthLive,
    queryFn: ({ signal }) => ohi.healthLive({ signal }),
    staleTime: 30 * 1000,
    refetchInterval: 30 * 1000,
    retry: false,
    ...options,
  });
}

export function useHealthReady(
  options?: Omit<UseQueryOptions<ReadinessStatus>, "queryKey" | "queryFn">,
) {
  return useQuery<ReadinessStatus>({
    queryKey: queryKeys.healthReady,
    queryFn: ({ signal }) => ohi.healthReady({ signal }),
    staleTime: 30 * 1000,
    refetchInterval: 30 * 1000,
    retry: false,
    ...options,
  });
}

export function useFeedbackMutation() {
  return useMutation({
    mutationFn: (req: FeedbackRequest) => ohi.feedback(req),
  });
}
