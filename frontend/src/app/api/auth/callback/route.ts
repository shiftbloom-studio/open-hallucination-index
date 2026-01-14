export const dynamic = 'force-dynamic';

import { createClient } from "@/lib/supabase/server";
import { NextResponse } from "next/server";

const INITIAL_TOKENS = 5;

export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);
  const code = searchParams.get("code");

  const next = searchParams.get("next");
  const safeNext = next?.startsWith("/") ? next : "/dashboard";

  if (code) {
    const supabase = await createClient();
    const { data, error } = await supabase.auth.exchangeCodeForSession(code);

    if (!error && data.user) {
      // Ensure user has a profile with initial tokens
      try {
        const { data: existingProfile } = await supabase
          .from('profiles')
          .select('id')
          .eq('id', data.user.id)
          .single();

        if (!existingProfile) {
          // Create new profile with initial tokens
          await supabase
            .from('profiles')
            .insert({
              id: data.user.id,
              email: data.user.email,
              tokens: INITIAL_TOKENS
            });
        }
      } catch (dbError) {
        console.error("Error creating user profile:", dbError);
        // Continue anyway - profile can be created on first API call
      }


      return NextResponse.redirect(`${origin}${safeNext}`);
    }
  }

  // return the user to an error page with instructions
  return NextResponse.redirect(`${origin}/auth/auth-code-error`);
}
