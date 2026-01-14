import { headers } from "next/headers";
import { NextResponse } from "next/server";
import Stripe from "stripe";
import { getStripe } from "@/lib/stripe";
import { createAdminClient } from "@/lib/supabase/admin";

export async function POST(req: Request) {
  const body = await req.text();
  // Await headers() in Next.js 15+, but likely this project is on 14/15. Wait, package.json said "next": "^16.1.1" so definitely await headers().
  const headerPayload = await headers();
  const signature = headerPayload.get("Stripe-Signature") as string;

  let event: Stripe.Event;

  try {
    event = getStripe().webhooks.constructEvent(
      body,
      signature,
      process.env.STRIPE_WEBHOOK_SECRET!
    );
  } catch (error) {
    console.error("[STRIPE_WEBHOOK] Error constructing event:", error);
    return new NextResponse("Webhook Error", {
      status: 400,
    });
  }

  const session = event.data.object as Stripe.Checkout.Session;

  if (event.type === "checkout.session.completed") {
    const userId = session.metadata?.userId;
    const packageId = session.metadata?.packageId;

    if (userId && packageId) {
      let tokensToAdd = 0;
      switch (packageId) {
        case "10":
          tokensToAdd = 10;
          break;
        case "100":
          tokensToAdd = 100;
          break;
        case "500":
          tokensToAdd = 500;
          break;
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
    }
  } else if (event.type === "checkout.session.expired") {
    console.warn("[STRIPE_WEBHOOK] Checkout session expired:", session.id);
  } else if (event.type === "checkout.session.async_payment_failed") {
    console.warn("[STRIPE_WEBHOOK] Async payment failed:", session.id);
  } else if (event.type === "payment_intent.payment_failed") {
    const intent = event.data.object as Stripe.PaymentIntent;
    console.warn("[STRIPE_WEBHOOK] Payment intent failed:", intent.id);
  }

  return new NextResponse(null, { status: 200 });
}
