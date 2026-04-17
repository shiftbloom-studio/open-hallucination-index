import type { CalibrationReport } from "@/lib/ohi-types";
import { cn } from "@/lib/utils";

export interface CalibrationTableProps {
  report: CalibrationReport;
  className?: string;
}

function coverageColor(empirical: number, target: number): string {
  const delta = Math.abs(empirical - target);
  if (delta <= 0.02) return "text-emerald-300";
  if (delta <= 0.05) return "text-amber-300";
  return "text-rose-300";
}

export function CalibrationTable({ report, className }: CalibrationTableProps) {
  const rows = Object.entries(report.domains);
  return (
    <div className={cn("overflow-x-auto rounded-xl border border-white/10 bg-white/[0.03]", className)}>
      <table
        className="w-full min-w-[560px] border-collapse text-left text-xs text-slate-300"
        data-testid="calibration-table"
      >
        <thead>
          <tr className="border-b border-white/10 text-[10px] uppercase tracking-wider text-slate-500">
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
              <tr key={name} className="border-b border-white/5 last:border-b-0">
                <td className="px-4 py-2 font-semibold text-slate-200">{name}</td>
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
      <footer className="border-t border-white/10 px-4 py-2 text-[10px] text-slate-500">
        Report date: <span className="font-mono text-slate-400">{report.report_date}</span> · target:{" "}
        <span className="font-mono text-slate-400">
          {Math.round(report.global_coverage_target * 100)}%
        </span>
      </footer>
    </div>
  );
}
