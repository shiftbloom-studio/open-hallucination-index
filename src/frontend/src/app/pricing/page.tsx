"use client";

import { motion } from "framer-motion";
import { Check, Sparkles, Zap, Crown, Shield, Lock, RefreshCw, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import Link from "next/link";
import { toast } from "sonner";
import { createClient } from "@/lib/supabase/client";
import { useState } from "react";

const packages = [
  {
    id: "10",
    name: "Starter",
    tokens: 10,
    price: "1.49",
    priceId: "price_1Smg720Fe33yJBCMhA1J38L9",
    features: [
      "10 OHI Tokens",
      "Up to 1,000 characters verified",
      "Full API access",
      "Tokens never expire",
      "Priority email support",
    ],
    icon: Sparkles,
    popular: false,
  },
  {
    id: "100",
    name: "Professional",
    tokens: 100,
    price: "9.99",
    priceId: "price_1Smg720Fe33yJBCMTj24emv8",
    features: [
      "100 OHI Tokens",
      "Up to 10,000 characters verified",
      "Full API access",
      "Tokens never expire",
      "Priority email support",
      "Advanced analytics",
    ],
    icon: Zap,
    popular: true,
    savings: 33,
  },
  {
    id: "500",
    name: "Enterprise",
    tokens: 500,
    price: "24.99",
    priceId: "price_1Smg720Fe33yJBCMo2AsQSxv",
    features: [
      "500 OHI Tokens",
      "Up to 50,000 characters verified",
      "Full API access",
      "Tokens never expire",
      "Priority email support",
      "Advanced analytics",
      "Dedicated account manager",
    ],
    icon: Crown,
    popular: false,
    savings: 67,
  },
];

const faqData = [
  {
    question: "What is an OHI Token?",
    answer:
      "An OHI Token allows you to verify up to 100 characters of text using our hallucination detection system. Each verification request consumes tokens based on the content length. Expert mode (with extensive evidence sources) costs 2 tokens per 100 characters.",
  },
  {
    question: "Do tokens expire?",
    answer:
      "No! All purchased tokens never expire. Use them at your own pace without any time restrictions.",
  },
  {
    question: "What payment methods do you accept?",
    answer:
      "We accept all major credit cards (Visa, Mastercard, American Express) through our secure Stripe payment gateway.",
  },
  {
    question: "How do I track my token usage?",
    answer:
      "Your current token balance is always visible in your dashboard. You can also view detailed usage history and analytics in your account settings.",
  },
  {
    question: "What happens if I run out of tokens?",
    answer:
      "You can purchase more tokens at any time. We also provide 2 free tokens daily to all registered users. Simply return to the pricing page to top up your balance.",
  },
];

function FAQItem({ question, answer }: { question: string; answer: string }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="border border-slate-800 rounded-lg bg-slate-900/50 overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-slate-800/50 transition-colors"
      >
        <span className="text-white font-medium">{question}</span>
        <ChevronDown
          className={`h-5 w-5 text-neutral-400 transition-transform duration-200 ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>
      {isOpen && (
        <div className="px-6 pb-4 text-neutral-300">
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}

export default function PricingPage() {
  const [loadingPackage, setLoadingPackage] = useState<string | null>(null);
  const [termsAccepted, setTermsAccepted] = useState<Record<string, boolean>>({});

  const handlePurchase = async (packageId: string): Promise<void> => {
    if (!termsAccepted[packageId]) {
      toast.error("Please accept the Terms of Service and AGB before purchasing");
      return;
    }

    setLoadingPackage(packageId);
    try {
      const supabase = createClient();
      const { data: { user } } = await supabase.auth.getUser();

      if (!user) {
        toast.error("Please sign in to purchase tokens");
        window.location.href = "/auth/login?redirect=/pricing";
        return;
      }

      const response = await fetch("/api/checkout", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          packageId,
          userId: user.id,
          userEmail: user.email,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to create checkout session");
      }

      const { url } = await response.json();
      window.location.href = url;
    } catch (error) {
      console.error("Checkout error:", error);
      toast.error("Failed to start checkout. Please try again.");
    } finally {
      setLoadingPackage(null);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">
      {/* Hero Section */}
      <section className="container mx-auto px-4 pt-20 pb-12 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Badge className="mb-4 bg-blue-500/10 text-blue-400 border-blue-500/20">
            Simple, Transparent Pricing
          </Badge>
          <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            Choose Your Plan
          </h1>
          <p className="text-xl text-neutral-300 max-w-2xl mx-auto mb-8">
            Pay only for what you need. All packages include lifetime access with no expiration.
          </p>
        </motion.div>
      </section>

      {/* Pricing Cards */}
      <section className="container mx-auto px-4 pb-20">
        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {packages.map((pkg, index) => {
            const Icon = pkg.icon;
            return (
              <motion.div
                key={pkg.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
              >
                <Card
                  className={`relative border-2 ${
                    pkg.popular
                      ? "border-blue-500 shadow-xl shadow-blue-500/20"
                      : "border-slate-800"
                  } bg-slate-900/50 backdrop-blur-xl hover:border-blue-500/50 transition-all duration-300 h-full flex flex-col`}
                >
                  {pkg.popular && (
                    <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                      <Badge className="bg-gradient-to-r from-blue-500 to-purple-500 text-white border-0">
                        Most Popular
                      </Badge>
                    </div>
                  )}
                  {pkg.savings && (
                    <div className="absolute -top-4 right-4">
                      <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
                        Save {pkg.savings}%
                      </Badge>
                    </div>
                  )}

                  <CardHeader className="text-center pb-4">
                    <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br from-blue-500/20 to-purple-500/20 border border-blue-500/30">
                      <Icon className="h-8 w-8 text-blue-400" />
                    </div>
                    <CardTitle className="text-2xl text-white">{pkg.name}</CardTitle>
                    <div className="mt-4">
                      <span className="text-5xl font-bold text-white">{pkg.price}€</span>
                    </div>
                    <CardDescription className="text-neutral-400 mt-2">
                      {pkg.tokens} tokens
                    </CardDescription>
                  </CardHeader>

                  <CardContent className="flex-1">
                    <ul className="space-y-3">
                      {pkg.features.map((feature, i) => (
                        <li key={i} className="flex items-start gap-3">
                          <Check className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />
                          <span className="text-neutral-300">{feature}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>

                  <CardFooter className="pt-4 flex-col gap-4">
                    <div className="flex items-start gap-2 w-full">
                      <Checkbox
                        id={`terms-${pkg.id}`}
                        checked={termsAccepted[pkg.id] || false}
                        onCheckedChange={(checked: boolean) => 
                          setTermsAccepted((prev) => ({ ...prev, [pkg.id]: checked === true }))
                        }
                        className="mt-1"
                      />
                      <label
                        htmlFor={`terms-${pkg.id}`}
                        className="text-sm text-neutral-400 leading-tight cursor-pointer"
                      >
                        I accept the{" "}
                        <Link href="/agb" className="text-blue-400 hover:underline" target="_blank">
                          Terms of Service (AGB)
                        </Link>
                        {" "}and{" "}
                        <Link href="/datenschutz" className="text-blue-400 hover:underline" target="_blank">
                          Privacy Policy
                        </Link>
                      </label>
                    </div>
                    <Button
                      onClick={() => handlePurchase(pkg.id)}
                      disabled={loadingPackage === pkg.id || !termsAccepted[pkg.id]}
                      className={`w-full font-semibold ${
                        pkg.popular
                          ? "bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600"
                          : "bg-slate-800 hover:bg-slate-700"
                      } text-white disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      {loadingPackage === pkg.id ? "Processing..." : "Buy Now"}
                    </Button>
                  </CardFooter>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* Trust Badges */}
      <section className="container mx-auto px-4 pb-20">
        <div className="max-w-4xl mx-auto">
          <div className="grid md:grid-cols-3 gap-8">
            <div className="flex flex-col items-center text-center p-6 rounded-xl bg-slate-900/50 border border-slate-800">
              <Shield className="h-12 w-12 text-blue-400 mb-4" />
              <h3 className="text-lg font-semibold text-white mb-2">GDPR Compliant</h3>
              <p className="text-sm text-neutral-400">
                Your data is protected under EU regulations
              </p>
            </div>
            <div className="flex flex-col items-center text-center p-6 rounded-xl bg-slate-900/50 border border-slate-800">
              <Lock className="h-12 w-12 text-green-400 mb-4" />
              <h3 className="text-lg font-semibold text-white mb-2">SSL Secured</h3>
              <p className="text-sm text-neutral-400">
                256-bit encryption for all transactions
              </p>
            </div>
            <div className="flex flex-col items-center text-center p-6 rounded-xl bg-slate-900/50 border border-slate-800">
              <RefreshCw className="h-12 w-12 text-purple-400 mb-4" />
              <h3 className="text-lg font-semibold text-white mb-2">Fair Terms</h3>
              <p className="text-sm text-neutral-400">
                Transparent terms of service
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="container mx-auto px-4 pb-20">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-8 text-white">
            Frequently Asked Questions
          </h2>
          <div className="space-y-4">
            {faqData.map((faq, index) => (
              <FAQItem key={index} question={faq.question} answer={faq.answer} />
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-4 pb-20">
        <div className="max-w-4xl mx-auto text-center p-12 rounded-2xl bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/20">
          <h2 className="text-3xl font-bold text-white mb-4">Still have questions?</h2>
          <p className="text-neutral-300 mb-8">
            Our team is here to help. Get in touch and we'll respond within 24 hours.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              asChild
              className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white font-semibold"
            >
              <Link href="mailto:hi@shiftbloom.studio">Contact Support</Link>
            </Button>
            <Button asChild variant="outline" className="border-slate-700 hover:border-slate-600">
              <Link href="/dashboard">Go to Dashboard</Link>
            </Button>
          </div>
        </div>
      </section>
    </main>
  );
}
