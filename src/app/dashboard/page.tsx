export const dynamic = 'force-dynamic';

import { Suspense } from "react";
import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import DashboardClient from "./dashboard-client";

export default async function DashboardPage() {
  const supabase = await createClient();

  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/auth/login");
  }

  return (
    <Suspense fallback={null}>
      <DashboardClient user={user} />
    </Suspense>
  );
}
