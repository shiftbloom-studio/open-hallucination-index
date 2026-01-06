"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Check, Loader2, Sparkles, Zap, Shield, Crown, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { createClient } from "@/lib/supabase/client";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

const packages = [
  {
    id: "10",
    tokens: 10,
    price: "1.49€",
    description: "Perfect for testing the waters.",
    icon: Sparkles,
    highlight: false,
    color: "from-blue-400 to-blue-600",
  },
  {
    id: "100",
    tokens: 100,
    price: "9.99€",
    description: "Our most popular choice for creators.",
    icon: Zap,
    highlight: false,
    color: "from-purple-400 to-purple-600",
  },
  {
    id: "500",
    tokens: 500,
    price: "24.99€",
    description: "Best value for serious hallucinations.",
    icon: Crown,
    highlight: true,
    tag: "Special Offer - Best Value",
    color: "from-amber-400 to-orange-600",
  },
];

export default function PricingPage() {
  const [loading, setLoading] = useState<string | null>(null);
  const [user, setUser] = useState<any>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [selectedPackage, setSelectedPackage] = useState<string | null>(null);
  
  // Auth Form State
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(false);

  const supabase = createClient();
  const router = useRouter();

  useEffect(() => {
    const getUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      setUser(user);
    };
    getUser();
  }, [supabase]);

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
    } catch (error) {
      toast.error("Something went wrong. Please try again.");
      setLoading(null);
    }
  };

  const handleAuthAndPurchase = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedPackage) return;
    setAuthLoading(true);

    try {
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
        // Proceed to checkout with the new user ID
        await handleCheckout(selectedPackage, data.user.id, email);
      }
    } catch (error) {
      toast.error("Authentication failed");
      setAuthLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 py-20 px-4 relative overflow-hidden">
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
                className="text-4xl md:text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400"
            >
                Get More OHI-Tokens
            </motion.h1>
            <motion.p 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto"
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
                    <li className="flex items-center text-sm text-slate-300">
                      <Check className="w-4 h-4 mr-2 text-green-500" />
                      Instant Access
                    </li>
                    <li className="flex items-center text-sm text-slate-300">
                      <Check className="w-4 h-4 mr-2 text-green-500" />
                      Secure Payment via Stripe
                    </li>
                    <li className="flex items-center text-sm text-slate-300">
                      <Check className="w-4 h-4 mr-2 text-green-500" />
                      No Expiration
                    </li>
                  </ul>
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
                        user ? "Buy Now" : "Sign up & Buy"
                    )}
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>
          ))}
        </div>
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
                    className="absolute top-4 right-4 text-slate-400 hover:text-white"
                >
                    <X className="w-5 h-5" />
                </button>
                <div className="mb-6">
                    <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400 mb-2">
                        Create Account to Purchase
                    </h2>
                    <p className="text-slate-400 text-sm">
                        Enter your details to create an account and proceed to secure checkout.
                    </p>
                </div>
                
                <form onSubmit={handleAuthAndPurchase} className="space-y-4">
                    <div className="space-y-2">
                    <Label htmlFor="email">Email</Label>
                    <Input
                        id="email"
                        type="email"
                        placeholder="you@example.com"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
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
                        className="bg-slate-950 border-slate-700 focus:border-blue-500 text-slate-100"
                    />
                    </div>
                    <Button type="submit" className="w-full bg-blue-600 hover:bg-blue-700" disabled={authLoading}>
                    {authLoading ? (
                        <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Creating Account...
                        </>
                    ) : (
                        "Continue to Payment"
                    )}
                    </Button>
                </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
