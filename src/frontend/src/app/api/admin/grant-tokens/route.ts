import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { email, tokens, reason } = body;

    if (!email || typeof email !== "string") {
      return NextResponse.json({ error: "Email is required" }, { status: 400 });
    }

    if (!tokens || typeof tokens !== "number" || tokens < 1) {
      return NextResponse.json({ error: "Valid token amount is required" }, { status: 400 });
    }

    const supabase = await createClient();

    // Verify the requesting user is authenticated
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Find the target user by email
    // RLS policy "admin_read_all_profiles" allows admins to read all rows
    const { data: targetUser, error: findError } = await supabase
      .from("profiles")
      .select("id, email, tokens")
      .eq("email", email)
      .single();

    if (findError || !targetUser) {
      // Could be RLS blocking or user not found
      if (findError?.code === "PGRST116") {
        return NextResponse.json({ error: "Admin access required or user not found" }, { status: 403 });
      }
      return NextResponse.json({ error: `User with email ${email} not found` }, { status: 404 });
    }

    // Grant tokens
    // RLS policy "admin_update_profiles" allows admins to update any profile
    const newTokens = (targetUser.tokens || 0) + tokens;
    const { error: updateError } = await supabase
      .from("profiles")
      .update({ 
        tokens: newTokens, 
        updated_at: new Date().toISOString() 
      })
      .eq("id", targetUser.id);

    if (updateError) {
      console.error("Error granting tokens:", updateError);
      if (updateError.code === "PGRST116" || updateError.message?.includes("permission")) {
        return NextResponse.json({ error: "Admin access required" }, { status: 403 });
      }
      return NextResponse.json({ error: "Failed to grant tokens" }, { status: 500 });
    }

    return NextResponse.json({
      success: true,
      message: `Granted ${tokens} tokens to ${email} (reason: ${reason || "Admin grant"})`,
      user_email: email,
      tokens_granted: tokens,
      new_balance: newTokens,
    });
  } catch (error) {
    console.error("Error in grant tokens route:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
