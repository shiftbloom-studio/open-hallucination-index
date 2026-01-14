import { NextResponse } from 'next/server';
import { getStripe } from '@/lib/stripe';

export async function POST(req: Request) {
  try {
    const { packageId, userId, userEmail } = await req.json();

    if (!packageId) {
      return new NextResponse("Missing packageId", { status: 400 });
    }

    let priceId;

    switch (packageId) {
      case '10':
        priceId = 'price_1Smg720Fe33yJBCMhA1J38L9'; // 1.49 EUR
        break;
      case '100':
        priceId = 'price_1Smg720Fe33yJBCMTj24emv8'; // 9.99 EUR
        break;
      case '500':
        priceId = 'price_1Smg720Fe33yJBCMo2AsQSxv'; // 24.99 EUR
        break;
      default:
        return new NextResponse("Invalid packageId", { status: 400 });
    }

    const session = await getStripe().checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [
        {
          price: priceId,
          quantity: 1,
        },
      ],
      mode: 'payment',
      success_url: `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/dashboard?success=true&session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/pricing?canceled=true&session_id={CHECKOUT_SESSION_ID}`,
      metadata: {
        userId: userId,
        packageId: packageId,
      },
      customer_email: userEmail,
    });

    return NextResponse.json({ url: session.url });
  } catch (error) {
    console.error('[STRIPE_CHECKOUT]', error);
    return new NextResponse("Internal Error", { status: 500 });
  }
}
