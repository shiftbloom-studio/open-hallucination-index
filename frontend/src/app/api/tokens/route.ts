export const dynamic = "force-dynamic";

import { createClient } from "@/lib/supabase/server";
import { NextResponse } from "next/server";

const INITIAL_TOKENS = 5;
const DAILY_TOKENS = 5;

// Helper function to check if a day has passed since last claim
function canClaimDailyTokens(lastClaimDate: string | null): boolean {
  if (!lastClaimDate) return true;
  
  const lastClaim = new Date(lastClaimDate);
  const now = new Date();
  
  // Reset at midnight UTC
  const lastClaimDay = new Date(lastClaim.getFullYear(), lastClaim.getMonth(), lastClaim.getDate());
  const todayDay = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  
  return todayDay > lastClaimDay;
}

// GET /api/tokens - Get current user's token balance
export async function GET() {
  try {
    const supabase = await createClient();
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser();

    if (authError || !user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Get or create user profile in Supabase
    const { data: fetchedProfile, error: profileError } = await supabase
      .from("profiles")
      .select("tokens, last_daily_claim")
      .eq("id", user.id)
      .single();

    let profile = fetchedProfile;

    if (profileError && profileError.code === "PGRST116") {
      // Profile doesn't exist, create it with initial tokens
      const { data: newProfile, error: createError } = await supabase
        .from("profiles")
        .insert({ 
          id: user.id, 
          email: user.email, 
          tokens: INITIAL_TOKENS,
          last_daily_claim: new Date().toISOString()
        })
        .select("tokens, last_daily_claim")
        .single();

      if (createError) {
        console.error("Error creating profile:", createError);
        return NextResponse.json(
          { error: "Failed to create profile" },
          { status: 500 }
        );
      }
      profile = newProfile;
    } else if (profileError) {
      console.error("Error fetching profile:", profileError);
      return NextResponse.json(
        { error: "Failed to fetch profile" },
        { status: 500 }
      );
    }

    // Check if user can claim daily tokens
    if (profile && canClaimDailyTokens(profile.last_daily_claim)) {
      const newTokenBalance = (profile.tokens ?? 0) + DAILY_TOKENS;
      
      const { data: updatedProfile, error: updateError } = await supabase
        .from("profiles")
        .update({ 
          tokens: newTokenBalance,
          last_daily_claim: new Date().toISOString()
        })
        .eq("id", user.id)
        .select("tokens, last_daily_claim")
        .single();

      if (!updateError && updatedProfile) {
        profile = updatedProfile;
      }
    }

    return NextResponse.json({
      tokens: profile?.tokens ?? INITIAL_TOKENS,
      email: user.email,
    });
  } catch (error) {
    console.error("Error fetching tokens:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

// POST /api/tokens - Deduct tokens based on actual content length
export async function POST(request: Request) {
  try {
    const supabase = await createClient();
    const {
      data: { user },
      error: authError,
    } = await supabase.auth.getUser();

    if (authError || !user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { text, context } = await request.json();

    if (typeof text !== "string") {
      return NextResponse.json(
        { error: "Text is required and must be a string" },
        { status: 400 }
      );
    }

    // Calculate tokens needed (1 token per 1000 characters, minimum 1)
    // We derive this from the actual strings, not a self-reported length
    const totalLength = text.length + (context?.length ?? 0);
    const tokensNeeded = Math.max(1, Math.ceil(totalLength / 1000));

    // Atomic deduction using RPC to prevent race conditions and ensure non-negative balance
    const { data, error: rpcError } = await supabase.rpc(
      "deduct_profile_tokens",
      {
        p_profile_id: user.id,
        p_amount: tokensNeeded,
      }
    );

    if (rpcError) {
      console.error("RPC Error (deduct_profile_tokens):", rpcError);

      // Specifically handle the "Insufficient tokens" case if the RPC raises an exception or returns null
      // Assuming RPC returns updated tokens or null if failed
      return NextResponse.json(
        {
          error: "Insufficient tokens or transaction failed",
          tokensNeeded,
        },
        { status: 402 }
      );
    }

    // If data is null or undefined, the RPC might have returned nothing because the 'where' clause failed
    if (data === null) {
      return NextResponse.json(
        {
          error: "Insufficient tokens",
          tokensNeeded,
        },
        { status: 402 }
      );
    }

    return NextResponse.json({
      success: true,
      tokensDeducted: tokensNeeded,
      tokensRemaining: data, // RPC returns the new balance
    });
  } catch (error) {
    console.error("Error deducting tokens:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
