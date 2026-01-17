/**
 * Supabase Admin Client - SERVICE ROLE ONLY
 * 
 * ⚠️ WARNING: This client bypasses RLS policies.
 * ONLY use for server-to-server operations where no user context exists:
 * - Stripe webhook handlers
 * - Cron jobs
 * - System-level operations
 * 
 * For user-facing admin operations, use the regular server client
 * with RLS policies that check the user's role.
 */

import { createClient as createSupabaseClient } from "@supabase/supabase-js";

export function createAdminClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !serviceRoleKey) {
    throw new Error(
      "Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables"
    );
  }

  return createSupabaseClient(supabaseUrl, serviceRoleKey, {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
  });
}
