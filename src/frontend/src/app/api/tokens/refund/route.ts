export const dynamic = "force-dynamic";

import { createClient } from "@/lib/supabase/server";
import { NextResponse } from "next/server";

// POST /api/tokens/refund - Refund tokens to user (e.g., when verification fails)
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

    const { amount } = await request.json();

    if (typeof amount !== "number" || amount <= 0) {
      return NextResponse.json(
        { error: "Amount must be a positive number" },
        { status: 400 }
      );
    }

    // Get current token balance
    const { data: profile, error: fetchError } = await supabase
      .from("profiles")
      .select("tokens")
      .eq("id", user.id)
      .single();

    if (fetchError) {
      console.error("Error fetching profile for refund:", fetchError);
      return NextResponse.json(
        { error: "Failed to fetch profile" },
        { status: 500 }
      );
    }

    const currentTokens = profile?.tokens ?? 0;
    const newBalance = currentTokens + amount;

    // Update token balance
    const { error: updateError } = await supabase
      .from("profiles")
      .update({ tokens: newBalance })
      .eq("id", user.id);

    if (updateError) {
      console.error("Error updating tokens for refund:", updateError);
      return NextResponse.json(
        { error: "Failed to refund tokens" },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      tokensRefunded: amount,
      tokensRemaining: newBalance,
    });
  } catch (error) {
    console.error("Error refunding tokens:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
