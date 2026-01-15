import { headers } from "next/headers";
import { NextResponse } from "next/server";
import Stripe from "stripe";
import { getStripe } from "@/lib/stripe";
import { createAdminClient } from "@/lib/supabase/admin";

const PACKAGE_TOKENS: Record<string, number> = {
  "10": 10,
  "100": 100,
  "500": 500,
};

function isValidUuid(value: string): boolean {
  // Simple UUID v4 format check; adjust if your IDs use a different format.
  const uuidRegex =
    /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  return uuidRegex.test(value);
}

export async function POST(req: Request) {
  const body = await req.text();
  // headers() is asynchronous in current Next.js versions and must be awaited.
  const headerPayload = await headers();
  const signature = headerPayload.get("Stripe-Signature") as string;

  let event: Stripe.Event;

  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!webhookSecret) {
    console.error("[STRIPE_WEBHOOK] Missing STRIPE_WEBHOOK_SECRET environment variable.");
    return new NextResponse("Server configuration error", {
      status: 500,
    });
  }

  try {
    event = getStripe().webhooks.constructEvent(
      body,
      signature,
      webhookSecret
    );
  } catch (error) {
    console.error("[STRIPE_WEBHOOK] Error constructing event:", error);
    return new NextResponse("Webhook Error", {
      status: 400,
    });
  }

  if (event.type === "checkout.session.completed") {
    const session = event.data.object as Stripe.Checkout.Session;
    const userId = session.metadata?.userId;
    const packageId = session.metadata?.packageId;

    if (!userId || !packageId) {
      console.warn(
        "[STRIPE_WEBHOOK] Missing userId or packageId in session metadata:",
        { sessionId: session.id, userId, packageId }
      );
      return new NextResponse("Invalid metadata", { status: 400 });
    }

    if (!isValidUuid(userId)) {
      console.warn(
        "[STRIPE_WEBHOOK] Invalid userId format in session metadata:",
        { sessionId: session.id, userId }
      );
      return new NextResponse("Invalid metadata", { status: 400 });
    }

    const tokensToAdd = PACKAGE_TOKENS[packageId];
    if (tokensToAdd === undefined) {
      console.warn(
        "[STRIPE_WEBHOOK] Unexpected packageId in session metadata:",
        { sessionId: session.id, packageId }
      );
      return new NextResponse("Invalid metadata", { status: 400 });
    }

    if (tokensToAdd > 0) {
      try {
        const supabase = createAdminClient();
        const { error } = await supabase.rpc("increment_profile_tokens", {
          profile_id: userId,
          token_amount: tokensToAdd,
        });

        if (error) {
          console.error("[STRIPE_WEBHOOK] Supabase update failed:", error);
          return new NextResponse("Database Error", { status: 500 });
        }

        console.log(
          `[STRIPE_WEBHOOK] Added ${tokensToAdd} tokens to user ${userId}`
        );
      } catch (error) {
        console.error("[STRIPE_WEBHOOK] Database update failed:", error);
        return new NextResponse("Database Error", { status: 500 });
      }
    }
  } else if (event.type === "checkout.session.expired") {
    const session = event.data.object as Stripe.Checkout.Session;
    console.warn("[STRIPE_WEBHOOK] Checkout session expired:", session.id);
  } else if (event.type === "checkout.session.async_payment_failed") {
    const session = event.data.object as Stripe.Checkout.Session;
    console.warn("[STRIPE_WEBHOOK] Async payment failed:", session.id);
  } else if (event.type === "payment_intent.payment_failed") {
    const intent = event.data.object as Stripe.PaymentIntent;
    console.warn("[STRIPE_WEBHOOK] Payment intent failed:", intent.id);
  }

  return new NextResponse(null, { status: 200 });
}
