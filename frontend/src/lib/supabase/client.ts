import { createBrowserClient } from '@supabase/ssr';

function createMockClient() {
  return {
    auth: {
      getUser: async () => ({ data: { user: null }, error: null }),
      onAuthStateChange: () => ({
        data: { subscription: { unsubscribe: () => undefined } },
      }),
      signOut: async () => ({ error: null }),
      signInWithPassword: async () => ({
        data: { user: null, session: null },
        error: { message: 'Supabase not configured' },
      }),
      signUp: async () => ({
        data: { user: null, session: null },
        error: { message: 'Supabase not configured' },
      }),
    },
  } as ReturnType<typeof createBrowserClient>;
}

export function createClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseAnonKey) {
    return createMockClient();
  }

  return createBrowserClient(supabaseUrl, supabaseAnonKey);
}
