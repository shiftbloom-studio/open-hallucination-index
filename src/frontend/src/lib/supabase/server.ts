import { createServerClient, type CookieOptions } from '@supabase/ssr';
import { cookies } from 'next/headers';

function createMockClient() {
  return {
    auth: {
      getUser: async () => ({ data: { user: null }, error: null }),
      exchangeCodeForSession: async () => ({
        data: { session: null, user: null },
        error: { message: 'Supabase not configured' },
      }),
    },
    from: () => ({
      select: () => ({
        eq: () => ({
          single: async () => ({
            data: null,
            error: { message: 'Supabase not configured', code: 'SUPABASE_NOT_CONFIGURED' },
          }),
        }),
      }),
      insert: () => ({
        select: () => ({
          single: async () => ({
            data: null,
            error: { message: 'Supabase not configured' },
          }),
        }),
      }),
      update: () => ({
        eq: async () => ({
          error: { message: 'Supabase not configured' },
        }),
      }),
    }),
  } as ReturnType<typeof createServerClient>;
}

export async function createClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseAnonKey) {
    return createMockClient();
  }

  const cookieStore = await cookies();

  return createServerClient(supabaseUrl, supabaseAnonKey, {
    cookies: {
      getAll() {
        return cookieStore.getAll();
      },
      setAll(cookiesToSet: Array<{ name: string; value: string; options?: CookieOptions }>) {
        try {
          cookiesToSet.forEach(({ name, value, options }) =>
            cookieStore.set(name, value, options)
          );
        } catch {
          // The `setAll` method was called from a Server Component.
          // This can be ignored if you have middleware refreshing
          // user sessions.
        }
      },
    },
  });
}
