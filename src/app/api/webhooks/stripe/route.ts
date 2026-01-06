import { headers } from 'next/headers';
import { NextResponse } from 'next/server';
import Stripe from 'stripe';
import { stripe } from '@/lib/stripe';
import { db } from '@/lib/db';
import { users } from '@/lib/db/schema';
import { eq, sql } from 'drizzle-orm';

export async function POST(req: Request) {
  const body = await req.text();
  // Await headers() in Next.js 15+, but likely this project is on 14/15. Wait, package.json said "next": "^16.1.1" so definitely await headers().
  const headerPayload = await headers();
  const signature = headerPayload.get("Stripe-Signature") as string;

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(
      body,
      signature,
      process.env.STRIPE_WEBHOOK_SECRET!
    );
  } catch (error: any) {
    return new NextResponse(`Webhook Error: ${error.message}`, { status: 400 });
  }

  const session = event.data.object as Stripe.Checkout.Session;

  if (event.type === 'checkout.session.completed') {
    const userId = session.metadata?.userId;
    const packageId = session.metadata?.packageId;

    if (userId && packageId) {
      let tokensToAdd = 0;
      switch (packageId) {
        case '10': tokensToAdd = 10; break;
        case '100': tokensToAdd = 100; break;
        case '500': tokensToAdd = 500; break;
      }

      if (tokensToAdd > 0) {
        try {
          await db.update(users)
            .set({
              ohiTokens: sql`${users.ohiTokens} + ${tokensToAdd}`,
            })
            .where(eq(users.id, userId));
          
          console.log(`[STRIPE_WEBHOOK] Added ${tokensToAdd} tokens to user ${userId}`);
        } catch (error) {
           console.error('[STRIPE_WEBHOOK] Database update failed:', error);
           return new NextResponse("Database Error", { status: 500 });
        }
      }
    }
  }

  return new NextResponse(null, { status: 200 });
}
