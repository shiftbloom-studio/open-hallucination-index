import type { CalibrationReport } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";

export interface CalibrationTableProps {
  report: CalibrationReport;
  className?: string;
}

function coverageColor(empirical: number, target: number): string {
  const delta = Math.abs(empirical - target);
  if (delta <= 0.02) return "text-[color:var(--brand-success)]";
  if (delta <= 0.05) return "text-[color:var(--brand-warning)]";
  return "text-[color:var(--brand-danger)]";
}

export function CalibrationTable({ report, className }: CalibrationTableProps) {
  const rows = Object.entries(report.domains);
  return (
    <div className={cn("sb-panel overflow-x-auto", className)}>
      <table
        className="w-full min-w-[560px] border-collapse text-left text-xs text-brand-muted"
        data-testid="calibration-table"
      >
        <thead>
          <tr className="border-b border-[color:var(--border-subtle)] text-[10px] uppercase tracking-wider text-brand-subtle">
            <th className="px-4 py-2">Domain</th>
            <th className="px-3 py-2 text-right">calib n</th>
            <th className="px-3 py-2 text-right">empirical coverage</th>
            <th className="px-3 py-2 text-right">interval p50</th>
            <th className="px-3 py-2 text-right">interval p95</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={5} className="px-4 py-6 text-center text-slate-500">
                No calibration data available yet.
              </td>
            </tr>
          ) : (
            rows.map(([name, d]) => (
              <tr key={name} className="border-b border-[color:var(--border-subtle)] last:border-b-0">
                <td className="px-4 py-2 font-semibold text-brand-ink">{name}</td>
                <td className="px-3 py-2 text-right font-mono">{d.calibration_n.toLocaleString()}</td>
                <td
                  className={cn(
                    "px-3 py-2 text-right font-mono",
                    coverageColor(d.empirical_coverage, report.global_coverage_target),
                  )}
                >
                  {(d.empirical_coverage * 100).toFixed(1)}%
                </td>
                <td className="px-3 py-2 text-right font-mono">{d.interval_width_p50.toFixed(2)}</td>
                <td className="px-3 py-2 text-right font-mono">{d.interval_width_p95.toFixed(2)}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
      <footer className="border-t border-[color:var(--border-subtle)] px-4 py-2 text-[10px] text-brand-subtle">
        Report date: <span className="font-mono text-brand-muted">{report.report_date}</span> · target:{" "}
        <span className="font-mono text-brand-muted">
          {Math.round(report.global_coverage_target * 100)}%
        </span>
      </footer>
    </div>
  );
}
