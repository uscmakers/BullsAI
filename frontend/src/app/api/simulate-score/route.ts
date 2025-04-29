import { supabase } from "@/lib/supabase";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const { player, score, round } = await request.json();

    const { data, error } = await supabase
      .from("dart_scores")
      .insert([{ player, score, round }])
      .select();

    if (error) {
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    return NextResponse.json({ success: true, data });
  } catch (error) {
    return NextResponse.json({ error: "Failed to add score" }, { status: 500 });
  }
}
