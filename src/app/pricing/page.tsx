"use client";

import { useState, useEffect, Suspense } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Check, Loader2, Sparkles, Zap, Shield, Crown, X, ChevronDown, Star, Lock, RefreshCw, Users, Award } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { createClient } from "@/lib/supabase/client";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { type User } from "@supabase/supabase-js";

const packages = [
  {
    id: "10",
    tokens: 10,
    price: "1.49€",
    pricePerToken: "0.15€",
    description: "Perfect for testing the waters.",
    icon: Sparkles,
    highlight: false,
    color: "from-blue-400 to-blue-600",
    features: ["10 verification requests", "Basic API access", "Email support"],
  },
  {
    id: "100",
    tokens: 100,
    price: "9.99€",
    pricePerToken: "0.10€",
    description: "Our most popular choice for creators.",
    icon: Zap,
    highlight: false,
    color: "from-purple-400 to-purple-600",
    features: ["100 verification requests", "Full API access", "Priority support", "Detailed analytics"],
  },
  {
    id: "500",
    tokens: 500,
    price: "24.99€",
    pricePerToken: "0.05€",
    description: "Best value for serious hallucinations.",
    icon: Crown,
    highlight: true,
    tag: "Best Value – Save 67%",
    color: "from-amber-400 to-orange-600",
    features: ["500 verification requests", "Full API access", "Priority support", "Detailed analytics", "Custom integrations", "Dedicated account manager"],
  },
];

const testimonials = [
  {
    name: "Sarah Chen",
    role: "AI Lead @ TechCorp",
    content: "OHI reduced our hallucination rate by 40% in production. The verification API is incredibly fast.",
    avatar: "SC",
  },
  {
    name: "Marcus Weber",
    role: "CTO @ DataFlow",
    content: "Finally, a tool that gives us measurable confidence in our LLM outputs. Game changer for compliance.",
    avatar: "MW",
  },
  {
    name: "Dr. Emily Foster",
    role: "Research Director",
    content: "The atomic verification approach is exactly what we needed for our medical AI applications.",
    avatar: "EF",
  },
];

const faqs = [
  {
    q: "What is an OHI Token?",
    a: "Each token represents one verification request. When you submit text to our API, we decompose it into claims, verify each against our knowledge graph, and return a trust score. One token = one complete verification.",
  },
  {
    q: "Do tokens expire?",
    a: "No! Your tokens never expire. Use them whenever you need – today, next month, or next year. They're yours forever.",
  },
  {
    q: "Can I get a refund?",
    a: "Yes. If you're not satisfied within 14 days of purchase and haven't used more than 10% of your tokens, we'll provide a full refund – no questions asked.",
  },
  {
    q: "Is there an API rate limit?",
    a: "The API supports up to 100 requests per minute for all plans. Need more? Contact us for enterprise options with custom rate limits.",
  },
  {
    q: "How accurate is the verification?",
    a: "Our verification system achieves 99% accuracy on benchmark datasets. We continuously update our knowledge graph to ensure the highest quality results.",
  },
];

const trustBadges = [
  { icon: Lock, label: "256-bit SSL Encryption" },
  { icon: Shield, label: "GDPR Compliant" },
  { icon: RefreshCw, label: "14-Day Money Back" },
  { icon: Users, label: "10,000+ Users" },
];

function FAQItem({ question, answer, index }: { question: string; answer: string; index: number }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 * index }}
      className="border border-slate-800 rounded-xl overflow-hidden bg-slate-900/50"
    >
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-5 text-left hover:bg-slate-800/50 transition-colors"
      >
        <span className="font-medium text-white">{question}</span>
        <motion.div
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown className="w-5 h-5 text-slate-400" />
        </motion.div>
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div className="px-5 pb-5 text-slate-400">
              {answer}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function PricingContent() {
  const [loading, setLoading] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [selectedPackage, setSelectedPackage] = useState<string | null>(null);
  
  // Auth Form State
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "signup">("login");

  const supabase = createClient();
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const getUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      setUser(user);
    };
    getUser();
  }, [supabase]);

  useEffect(() => {
    const canceled = searchParams.get("canceled");
    const sessionId = searchParams.get("session_id");

    if (!canceled) {
      return;
    }

    const checkStatus = async () => {
      if (!sessionId) {
        toast.error("Payment canceled. You can try again anytime.");
        router.replace("/pricing");
        return;
      }

      try {
        const res = await fetch(`/api/checkout/status?session_id=${sessionId}`);
        if (!res.ok) throw new Error("Failed to fetch checkout status");
        const data = await res.json();

        if (data.status === "expired" || data.paymentStatus === "unpaid") {
          toast.error("Payment was canceled or expired. Please try again.");
        } else {
          toast.error("Payment not completed. Please try again.");
        }
      } catch {
        console.error("Failed to verify checkout status");
        toast.error("We couldn't confirm the payment status. Please try again.");
      } finally {
        router.replace("/pricing");
      }
    };

    checkStatus();
  }, [router, searchParams]);

  const handlePurchaseClick = (pkgId: string) => {
    setSelectedPackage(pkgId);
    if (user) {
      handleCheckout(pkgId, user.id);
    } else {
      setShowAuthModal(true);
    }
  };

  const handleCheckout = async (pkgId: string, userId: string, emailParam?: string) => {
    setLoading(pkgId);
    try {
      const res = await fetch("/api/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          packageId: pkgId,
          userId: userId,
          userEmail: emailParam ||(user ? user.email : undefined),
        }),
      });

      if (!res.ok) throw new Error("Checkout failed");

      const { url } = await res.json();
      window.location.href = url;
      return true;
    } catch {
      toast.error("Something went wrong. Please try again.");
      setLoading(null);
      return false;
    }
  };

  const handleAuthAndPurchase = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedPackage || authLoading) return;
    setAuthLoading(true);

    try {
      if (authMode === "login") {
        // Attempt Login
        const { data, error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });

        if (error) {
          toast.error(error.message);
          setAuthLoading(false);
          return;
        }

        if (data.user) {
          setUser(data.user);
          toast.success("Logged in! Redirecting to payment...");
          const checkoutSuccess = await handleCheckout(selectedPackage, data.user.id, email);
          if (!checkoutSuccess) {
            setAuthLoading(false);
          }
        } else {
          toast.error("Login was successful, but we couldn't load your account. Please try again.");
          setAuthLoading(false);
        }
      } else {
        // Attempt Sign Up
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            emailRedirectTo: `${window.location.origin}/api/auth/callback`,
          }
        });

        if (error) {
          toast.error(error.message);
          setAuthLoading(false);
          return;
        }

        if (data.user) {
          toast.success("Account created! Redirecting to payment...");
          const checkoutSuccess = await handleCheckout(selectedPackage, data.user.id, email);
          if (!checkoutSuccess) {
            setAuthLoading(false);
          }
        } else {
          toast.success("Account created! Please check your email to confirm before purchasing.");
          setAuthLoading(false);
        }
      }
    } catch {
      toast.error("Authentication failed");
      setAuthLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 py-20 px-4 relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-purple-900/20 rounded-full blur-[100px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[600px] h-[600px] bg-blue-900/20 rounded-full blur-[100px]" />
      </div>

      <div className="relative z-10 max-w-6xl mx-auto">
        <div className="text-center mb-16 space-y-4">
            <motion.h1 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-4xl md:text-6xl lg:text-7xl font-heading font-bold tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400"
            >
                Get More OHI-Tokens
            </motion.h1>
            <motion.p 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="text-slate-400 text-lg md:text-xl lg:text-2xl max-w-2xl mx-auto font-light leading-relaxed tracking-wide"
            >
                Unlock the full potential of your hallucination index. Purchase tokens to verify and analyze more content.
            </motion.p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {packages.map((pkg, index) => (
            <motion.div
              key={pkg.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 + 0.2 }}
            >
              <Card className={cn(
                "relative h-full flex flex-col border-slate-800 bg-slate-900/50 backdrop-blur-xl transition-all duration-300 hover:border-slate-600 hover:shadow-2xl hover:shadow-purple-500/10",
                pkg.highlight && "border-amber-500/50 shadow-amber-500/10 scale-105 z-10"
              )}>
                {pkg.highlight && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 bg-gradient-to-r from-amber-500 to-orange-600 text-white px-4 py-1 rounded-full text-sm font-bold shadow-lg">
                    {pkg.tag}
                  </div>
                )}
                <CardHeader>
                    <div className={cn("w-12 h-12 rounded-lg flex items-center justify-center mb-4 bg-gradient-to-br", pkg.color)}>
                        <pkg.icon className="w-6 h-6 text-white" />
                    </div>
                  <CardTitle className="text-2xl">{pkg.tokens} Tokens</CardTitle>
                  <CardDescription className="text-slate-400">{pkg.description}</CardDescription>
                </CardHeader>
                <CardContent className="flex-grow">
                  <div className="text-4xl font-bold mb-6">
                    {pkg.price}
                    <span className="text-sm font-normal text-slate-500 ml-2">one-time</span>
                  </div>
                  <ul className="space-y-3">
                    {pkg.features.map((feature, i) => (
                      <li key={i} className="flex items-center text-sm text-slate-300">
                        <Check className="w-4 h-4 mr-2 text-green-500 flex-shrink-0" />
                        {feature}
                      </li>
                    ))}
                  </ul>
                  <div className="mt-4 pt-4 border-t border-slate-700">
                    <span className="text-xs text-slate-500">Price per token: </span>
                    <span className="text-sm font-semibold text-slate-300">{pkg.pricePerToken}</span>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button 
                    onClick={() => handlePurchaseClick(pkg.id)}
                    disabled={!!loading}
                    className={cn(
                        "w-full text-lg h-12 font-semibold transition-all",
                        pkg.highlight 
                            ? "bg-gradient-to-r from-amber-500 to-orange-600 hover:from-amber-600 hover:to-orange-700 text-white border-0"
                            : "bg-slate-800 hover:bg-slate-700 text-white"
                    )}
                  >
                    {loading === pkg.id ? (
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    ) : (
                        "Buy Now"
                    )}
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Trust Badges */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-16 flex flex-wrap justify-center gap-6 md:gap-10"
        >
          {trustBadges.map((badge, i) => (
            <div key={i} className="flex items-center gap-2 text-slate-400">
              <badge.icon className="w-5 h-5" />
              <span className="text-sm">{badge.label}</span>
            </div>
          ))}
        </motion.div>

        {/* Money-Back Guarantee Banner */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.7 }}
          className="mt-16 relative overflow-hidden rounded-2xl border border-emerald-500/30 bg-gradient-to-r from-emerald-900/20 via-emerald-800/10 to-emerald-900/20 p-8 text-center"
        >
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(16,185,129,0.15),transparent_70%)]" />
          <div className="relative z-10">
            <Award className="w-12 h-12 mx-auto mb-4 text-emerald-400" />
            <h3 className="text-2xl font-bold text-white mb-2">14-Day Money-Back Guarantee</h3>
            <p className="text-slate-300 max-w-lg mx-auto">
              Not satisfied? Get a full refund within 14 days if you&apos;ve used less than 10% of your tokens. No questions asked.
            </p>
          </div>
        </motion.div>

        {/* Testimonials */}
        <div className="mt-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold text-white mb-4">Trusted by AI Teams Worldwide</h2>
            <p className="text-slate-400">See what our customers are saying about OHI</p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            {testimonials.map((testimonial, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 + i * 0.1 }}
                whileHover={{ y: -4, transition: { duration: 0.2 } }}
              >
                <Card className="h-full border-slate-800 bg-slate-900/50 backdrop-blur">
                  <CardContent className="pt-6">
                    <div className="flex gap-1 mb-4">
                      {[...Array(5)].map((_, j) => (
                        <Star key={j} className="w-4 h-4 fill-amber-400 text-amber-400" />
                      ))}
                    </div>
                    <p className="text-slate-300 mb-6 italic">&quot;{testimonial.content}&quot;</p>
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center text-white font-semibold text-sm">
                        {testimonial.avatar}
                      </div>
                      <div>
                        <p className="font-semibold text-white">{testimonial.name}</p>
                        <p className="text-sm text-slate-400">{testimonial.role}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Stats Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-24 grid grid-cols-2 md:grid-cols-4 gap-6"
        >
          {[
            { value: "10M+", label: "Verifications Run" },
            { value: "99.2%", label: "Accuracy Rate" },
            { value: "<50ms", label: "Avg. Response Time" },
            { value: "4.9/5", label: "Customer Rating" },
          ].map((stat, i) => (
            <div key={i} className="text-center p-6 rounded-xl bg-slate-900/50 border border-slate-800">
              <p className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                {stat.value}
              </p>
              <p className="text-sm text-slate-400 mt-1">{stat.label}</p>
            </div>
          ))}
        </motion.div>

        {/* FAQ Section */}
        <div className="mt-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold text-white mb-4">Frequently Asked Questions</h2>
            <p className="text-slate-400">Everything you need to know about OHI tokens</p>
          </motion.div>

          <div className="max-w-3xl mx-auto space-y-4">
            {faqs.map((faq, i) => (
              <FAQItem key={i} question={faq.q} answer={faq.a} index={i} />
            ))}
          </div>
        </div>

        {/* Final CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-24 text-center"
        >
          <h2 className="text-3xl font-bold text-white mb-4">Ready to verify with confidence?</h2>
          <p className="text-slate-400 mb-8 max-w-lg mx-auto">
            Join thousands of teams using OHI to build trustworthy AI applications.
          </p>
          <Button
            onClick={() => document.querySelector('.grid.grid-cols-1')?.scrollIntoView({ behavior: 'smooth' })}
            className="bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white px-8 py-6 text-lg rounded-full"
          >
            Choose Your Plan
          </Button>
        </motion.div>
      </div>

      <AnimatePresence>
        {showAuthModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setShowAuthModal(false)}
                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            />
            <motion.div 
               initial={{ opacity: 0, scale: 0.95, y: 20 }}
               animate={{ opacity: 1, scale: 1, y: 0 }}
               exit={{ opacity: 0, scale: 0.95, y: 20 }}
               className="relative w-full max-w-md bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-2xl z-10"
            >
                <button 
                    onClick={() => setShowAuthModal(false)}
                    disabled={authLoading}
                    className="absolute top-4 right-4 text-slate-400 hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                    <X className="w-5 h-5" />
                </button>
                <div className="mb-6">
                    <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400 mb-2">
                        {authMode === "login" ? "Login to Continue" : "Create Account"}
                    </h2>
                    <p className="text-slate-400 text-sm">
                        {authMode === "login" 
                          ? "Log in to your account to proceed to checkout." 
                          : "Create an account to proceed to secure checkout."}
                    </p>
                </div>

                {/* Auth Mode Tabs */}
                <div className="flex mb-6 bg-slate-800 rounded-lg p-1">
                    <button
                        type="button"
                        onClick={() => setAuthMode("login")}
                        disabled={authLoading}
                        className={cn(
                            "flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all",
                            authMode === "login" 
                                ? "bg-blue-600 text-white" 
                                : "text-slate-400 hover:text-white",
                            authLoading && "opacity-60 cursor-not-allowed"
                        )}
                    >
                        Login
                    </button>
                    <button
                        type="button"
                        onClick={() => setAuthMode("signup")}
                        disabled={authLoading}
                        className={cn(
                            "flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all",
                            authMode === "signup" 
                                ? "bg-blue-600 text-white" 
                                : "text-slate-400 hover:text-white",
                            authLoading && "opacity-60 cursor-not-allowed"
                        )}
                    >
                        Sign Up
                    </button>
                </div>
                
                <form onSubmit={handleAuthAndPurchase} className="space-y-4" aria-busy={authLoading}>
                    <div className="space-y-2">
                    <Label htmlFor="email">Email</Label>
                    <Input
                        id="email"
                        type="email"
                        placeholder="you@example.com"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                        disabled={authLoading}
                        className="bg-slate-950 border-slate-700 focus:border-blue-500 text-slate-100"
                    />
                    </div>
                    <div className="space-y-2">
                    <Label htmlFor="password">Password</Label>
                    <Input
                        id="password"
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                        disabled={authLoading}
                        className="bg-slate-950 border-slate-700 focus:border-blue-500 text-slate-100"
                    />
                    </div>
                    <Button type="submit" className="w-full bg-blue-600 hover:bg-blue-700" disabled={authLoading}>
                    {authLoading ? (
                        <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" /> {authMode === "login" ? "Logging in..." : "Creating Account..."}
                        </>
                    ) : (
                        "Continue to Payment"
                    )}
                    </Button>
                    <p className="text-xs text-slate-500 text-center" aria-live="polite">
                      {authLoading ? "Processing your request. Please wait..." : "You can switch between login and sign up any time."}
                    </p>
                </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </main>
  );
}

export default function PricingPage() {
  return (
    <Suspense fallback={null}>
      <PricingContent />
    </Suspense>
  );
}
