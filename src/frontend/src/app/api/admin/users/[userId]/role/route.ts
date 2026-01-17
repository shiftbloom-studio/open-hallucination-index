import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ userId: string }> }
) {
  try {
    const { userId } = await params;
    const body = await request.json();
    const { role } = body;

    if (!role || !["user", "admin"].includes(role)) {
      return NextResponse.json({ error: "Invalid role" }, { status: 400 });
    }

    const supabase = await createClient();

    // Verify the requesting user is authenticated
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Prevent self-demotion
    if (userId === user.id && role !== "admin") {
      return NextResponse.json({ error: "Cannot demote yourself" }, { status: 400 });
    }

    // Update the user's role
    // RLS policy "admin_update_profiles" allows admins to update any profile
    const { data: updatedUser, error } = await supabase
      .from("profiles")
      .update({ role, updated_at: new Date().toISOString() })
      .eq("id", userId)
      .select("id, email, tokens, role, created_at, updated_at")
      .single();

    if (error) {
      console.error("Error updating user role:", error);
      // If RLS blocks the update, it typically returns no rows or a permission error
      if (error.code === "PGRST116" || error.message?.includes("permission")) {
        return NextResponse.json({ error: "Admin access required" }, { status: 403 });
      }
      return NextResponse.json({ error: "Failed to update user role" }, { status: 500 });
    }

    if (!updatedUser) {
      return NextResponse.json({ error: "Admin access required or user not found" }, { status: 403 });
    }

    return NextResponse.json({
      id: updatedUser.id,
      email: updatedUser.email || "",
      name: null,
      tokens: updatedUser.tokens || 0,
      role: updatedUser.role || "user",
      created_at: updatedUser.created_at,
      updated_at: updatedUser.updated_at,
    });
  } catch (error) {
    console.error("Error in admin user role route:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
