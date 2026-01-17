import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";

interface ProfileRow {
  id: string;
  email: string | null;
  tokens: number | null;
  role: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export async function GET() {
  try {
    const supabase = await createClient();

    // Verify the requesting user is authenticated
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Fetch all users from profiles table
    // RLS policy "admin_read_all_profiles" allows admins to read all rows
    // Non-admins will get empty result or only their own row based on RLS
    const { data: users, error } = await supabase
      .from("profiles")
      .select("id, email, tokens, role, created_at, updated_at")
      .order("created_at", { ascending: false });

    if (error) {
      console.error("Error fetching users:", error);
      return NextResponse.json({ error: "Failed to fetch users", details: error.message }, { status: 500 });
    }

    // Transform to match the expected format
    const transformedUsers = (users || []).map((u: ProfileRow) => ({
      id: u.id,
      email: u.email || "",
      name: null,
      tokens: u.tokens || 0,
      role: u.role || "user",
      created_at: u.created_at,
      updated_at: u.updated_at || u.created_at,
    }));

    return NextResponse.json({
      users: transformedUsers,
      total: transformedUsers.length,
      page: 1,
      page_size: transformedUsers.length,
      total_pages: 1,
    });
  } catch (error) {
    console.error("Error in admin users route:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
