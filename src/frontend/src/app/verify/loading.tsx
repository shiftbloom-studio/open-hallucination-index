export default function Loading() {
  return (
    <div className="mx-auto grid min-h-[60vh] w-full max-w-7xl grid-cols-1 gap-6 px-4 py-8 lg:grid-cols-[minmax(0,32rem)_minmax(0,1fr)]">
      <div className="h-64 animate-pulse rounded-xl border border-[color:var(--border-subtle)] bg-surface-elevated" />
      <div className="h-64 animate-pulse rounded-xl border border-[color:var(--border-subtle)] bg-surface-elevated" />
    </div>
  );
}
