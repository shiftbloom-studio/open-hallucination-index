export const dynamic = 'force-dynamic';

import { createClient } from "@/lib/supabase/server";
import { NextResponse } from "next/server";

const INITIAL_TOKENS = 5;

// GET /api/tokens - Get current user's token balance
export async function GET() {
  try {
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Get or create user profile in Supabase
    const { data: fetchedProfile, error: profileError } = await supabase
      .from('profiles')
      .select('tokens')
      .eq('id', user.id)
      .single();

    let profile = fetchedProfile;

    if (profileError && profileError.code === 'PGRST116') {
      // Profile doesn't exist, create it with initial tokens
      const { data: newProfile, error: createError } = await supabase
        .from('profiles')
        .insert({ id: user.id, email: user.email, tokens: INITIAL_TOKENS })
        .select('tokens')
        .single();

      if (createError) {
        console.error("Error creating profile:", createError);
        return NextResponse.json({ error: "Failed to create profile" }, { status: 500 });
      }
      profile = newProfile;
    } else if (profileError) {
      console.error("Error fetching profile:", profileError);
      return NextResponse.json({ error: "Failed to fetch profile" }, { status: 500 });
    }

    return NextResponse.json({ 
      tokens: profile?.tokens ?? INITIAL_TOKENS,
      email: user.email,
    });
  } catch (error) {
    console.error("Error fetching tokens:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}

// POST /api/tokens - Deduct tokens based on text length
export async function POST(request: Request) {
  try {
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json();
    const { textLength } = body;

    if (typeof textLength !== 'number' || textLength <= 0) {
      return NextResponse.json({ error: "Invalid text length" }, { status: 400 });
    }

    // Calculate tokens needed (1 token per 1000 characters, minimum 1)
    const tokensNeeded = Math.max(1, Math.ceil(textLength / 1000));

    // Get current token balance
    const { data: fetchedProfile, error: profileError } = await supabase
      .from('profiles')
      .select('tokens')
      .eq('id', user.id)
      .single();

    let profile = fetchedProfile;

    if (profileError && profileError.code === 'PGRST116') {
      // Create profile with initial tokens if it doesn't exist
      const { data: newProfile, error: createError } = await supabase
        .from('profiles')
        .insert({ id: user.id, email: user.email, tokens: INITIAL_TOKENS })
        .select('tokens')
        .single();

      if (createError) {
        return NextResponse.json({ error: "Failed to create profile" }, { status: 500 });
      }
      profile = newProfile;
    } else if (profileError) {
      return NextResponse.json({ error: "Failed to fetch profile" }, { status: 500 });
    }

    const currentTokens = profile?.tokens ?? 0;

    if (currentTokens < tokensNeeded) {
      return NextResponse.json({ 
        error: "Insufficient tokens",
        tokensNeeded,
        tokensAvailable: currentTokens,
      }, { status: 402 });
    }

    // Deduct tokens
    const newBalance = currentTokens - tokensNeeded;
    const { error: updateError } = await supabase
      .from('profiles')
      .update({ tokens: newBalance, updated_at: new Date().toISOString() })
      .eq('id', user.id);

    if (updateError) {
      console.error("Error updating tokens:", updateError);
      return NextResponse.json({ error: "Failed to update tokens" }, { status: 500 });
    }

    return NextResponse.json({ 
      success: true,
      tokensDeducted: tokensNeeded,
      tokensRemaining: newBalance,
    });
  } catch (error) {
    console.error("Error deducting tokens:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
