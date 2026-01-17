export const dynamic = 'force-dynamic';

import { Suspense } from "react";
import { createClient } from "@/lib/supabase/server";
import AdminDashboardClient from "./admin-client";

export default async function AdminPage() {
  const supabase = await createClient();

  const {
    data: { user },
  } = await supabase.auth.getUser();

  return (
    <Suspense fallback={<AdminLoadingSkeleton />}>
      <AdminDashboardClient user={user!} />
    </Suspense>
  );
}

function AdminLoadingSkeleton() {
  return (
    <div className="space-y-6">
      <div className="h-8 w-48 bg-muted animate-pulse rounded" />
      <div className="grid gap-4 md:grid-cols-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-32 bg-muted animate-pulse rounded-lg" />
        ))}
      </div>
      <div className="h-96 bg-muted animate-pulse rounded-lg" />
    </div>
  );
}
