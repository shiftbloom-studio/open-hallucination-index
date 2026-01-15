import 'server-only';
import Stripe from 'stripe';

let stripeClient: Stripe | null = null;

export function getStripe() {
  const apiKey = process.env.STRIPE_SECRET_KEY;

  if (!apiKey) {
    throw new Error('Missing STRIPE_SECRET_KEY');
  }

  if (!stripeClient) {
    stripeClient = new Stripe(apiKey, {
      typescript: true,
    });
  }

  return stripeClient;
}
