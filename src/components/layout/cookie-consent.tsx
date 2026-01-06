"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cookie, X, Settings, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import Link from "next/link";

interface CookiePreferences {
  necessary: boolean;
  analytics: boolean;
  functional: boolean;
}

const COOKIE_CONSENT_KEY = "ohi_cookie_consent";
const COOKIE_PREFERENCES_KEY = "ohi_cookie_preferences";

export function CookieConsent() {
  const [showBanner, setShowBanner] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [preferences, setPreferences] = useState<CookiePreferences>({
    necessary: true, // Always required
    analytics: false,
    functional: false,
  });

  useEffect(() => {
    // Check if user has already consented
    const consent = localStorage.getItem(COOKIE_CONSENT_KEY);
    if (!consent) {
      // Small delay to prevent flash on page load
      const timer = setTimeout(() => setShowBanner(true), 1000);
      return () => clearTimeout(timer);
    } else {
      // Load saved preferences
      const savedPreferences = localStorage.getItem(COOKIE_PREFERENCES_KEY);
      if (savedPreferences) {
        setPreferences(JSON.parse(savedPreferences));
      }
    }
  }, []);

  // Listen for event to open cookie settings
  useEffect(() => {
    const handleOpenSettings = () => {
      setShowBanner(true);
      setShowSettings(true);
    };

    window.addEventListener("openCookieSettings", handleOpenSettings);
    return () => {
      window.removeEventListener("openCookieSettings", handleOpenSettings);
    };
  }, []);

  const saveConsent = (acceptAll: boolean = false) => {
    const finalPreferences = acceptAll
      ? { necessary: true, analytics: true, functional: true }
      : preferences;

    localStorage.setItem(COOKIE_CONSENT_KEY, "true");
    localStorage.setItem(COOKIE_PREFERENCES_KEY, JSON.stringify(finalPreferences));
    setPreferences(finalPreferences);
    setShowBanner(false);
    setShowSettings(false);

    // Dispatch custom event for analytics initialization
    if (finalPreferences.analytics) {
      window.dispatchEvent(new CustomEvent("cookieConsentGranted", { detail: finalPreferences }));
    }
  };

  const rejectAll = () => {
    const minimalPreferences = { necessary: true, analytics: false, functional: false };
    localStorage.setItem(COOKIE_CONSENT_KEY, "true");
    localStorage.setItem(COOKIE_PREFERENCES_KEY, JSON.stringify(minimalPreferences));
    setPreferences(minimalPreferences);
    setShowBanner(false);
    setShowSettings(false);
  };

  const togglePreference = (key: keyof CookiePreferences) => {
    if (key === "necessary") return; // Cannot disable necessary cookies
    setPreferences((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <AnimatePresence>
      {showBanner && (
        <motion.div
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 100, opacity: 0 }}
          transition={{ type: "spring", damping: 25, stiffness: 200 }}
          className="fixed bottom-0 left-0 right-0 z-[100] p-4 md:p-6"
        >
          <div className="max-w-4xl mx-auto bg-slate-900/95 backdrop-blur-xl border border-slate-700 rounded-2xl shadow-2xl shadow-black/50 overflow-hidden">
            {!showSettings ? (
              // Main Banner
              <div className="p-6">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-slate-800 rounded-xl shrink-0">
                    <Cookie className="w-6 h-6 text-amber-400" />
                  </div>
                  <div className="flex-1 space-y-4">
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-2">
                        We Value Your Privacy
                      </h3>
                      <p className="text-sm text-slate-300 leading-relaxed">
                        We use cookies to enhance your browsing experience, provide personalized content, and analyze our traffic. 
                        By clicking &quot;Accept All&quot;, you consent to our use of cookies as described in our{" "}
                        <Link href="/datenschutz" className="text-blue-400 hover:underline">
                          Privacy Policy
                        </Link>
                        . You can customize your preferences or reject non-essential cookies.
                      </p>
                    </div>
                    
                    <div className="flex flex-wrap gap-3">
                      <Button
                        onClick={() => saveConsent(true)}
                        className="bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white font-medium px-6"
                      >
                        <Check className="w-4 h-4 mr-2" />
                        Accept All
                      </Button>
                      <Button
                        onClick={rejectAll}
                        variant="outline"
                        className="border-slate-600 text-slate-300 hover:bg-slate-800 hover:text-white"
                      >
                        Reject All
                      </Button>
                      <Button
                        onClick={() => setShowSettings(true)}
                        variant="ghost"
                        className="text-slate-400 hover:text-white hover:bg-slate-800"
                      >
                        <Settings className="w-4 h-4 mr-2" />
                        Customize
                      </Button>
                    </div>
                  </div>
                  <button
                    onClick={rejectAll}
                    className="p-2 text-slate-500 hover:text-white transition-colors shrink-0"
                    aria-label="Close cookie banner"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              </div>
            ) : (
              // Settings Panel
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-white">Cookie Preferences</h3>
                  <button
                    onClick={() => setShowSettings(false)}
                    className="p-2 text-slate-500 hover:text-white transition-colors"
                    aria-label="Close settings"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                <div className="space-y-4 mb-6">
                  {/* Necessary Cookies */}
                  <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <span className="text-white font-medium">Strictly Necessary</span>
                        <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">
                          Always Active
                        </span>
                      </div>
                      <div className="w-12 h-6 bg-emerald-500 rounded-full flex items-center justify-end px-1 cursor-not-allowed">
                        <div className="w-4 h-4 bg-white rounded-full shadow" />
                      </div>
                    </div>
                    <p className="text-sm text-slate-400">
                      These cookies are essential for the website to function properly. They enable basic functions like page navigation, secure login, and access to secure areas. The website cannot function properly without these cookies.
                    </p>
                  </div>

                  {/* Analytics Cookies */}
                  <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-white font-medium">Analytics & Performance</span>
                      <button
                        onClick={() => togglePreference("analytics")}
                        className={`w-12 h-6 rounded-full flex items-center px-1 transition-colors ${
                          preferences.analytics ? "bg-emerald-500 justify-end" : "bg-slate-600 justify-start"
                        }`}
                        role="switch"
                        aria-checked={preferences.analytics}
                        aria-label="Toggle analytics cookies"
                      >
                        <div className="w-4 h-4 bg-white rounded-full shadow" />
                      </button>
                    </div>
                    <p className="text-sm text-slate-400">
                      These cookies help us understand how visitors interact with our website. We use Vercel Analytics to collect anonymous, aggregated data about page views and user behavior. This helps us improve our website and services. No personal data is collected.
                    </p>
                  </div>

                  {/* Functional Cookies */}
                  <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-white font-medium">Functional</span>
                      <button
                        onClick={() => togglePreference("functional")}
                        className={`w-12 h-6 rounded-full flex items-center px-1 transition-colors ${
                          preferences.functional ? "bg-emerald-500 justify-end" : "bg-slate-600 justify-start"
                        }`}
                        role="switch"
                        aria-checked={preferences.functional}
                        aria-label="Toggle functional cookies"
                      >
                        <div className="w-4 h-4 bg-white rounded-full shadow" />
                      </button>
                    </div>
                    <p className="text-sm text-slate-400">
                      These cookies enable enhanced functionality and personalization, such as remembering your preferences, language settings, and providing personalized features. If you disable these, some features may not work as intended.
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-3 pt-4 border-t border-slate-700">
                  <Button
                    onClick={() => saveConsent(false)}
                    className="bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white font-medium px-6"
                  >
                    Save Preferences
                  </Button>
                  <Button
                    onClick={() => saveConsent(true)}
                    variant="outline"
                    className="border-slate-600 text-slate-300 hover:bg-slate-800 hover:text-white"
                  >
                    Accept All
                  </Button>
                  <Button
                    onClick={rejectAll}
                    variant="ghost"
                    className="text-slate-400 hover:text-white hover:bg-slate-800"
                  >
                    Reject All
                  </Button>
                </div>

                <p className="text-xs text-slate-500 mt-4">
                  For more information, please read our{" "}
                  <Link href="/datenschutz" className="text-blue-400 hover:underline">
                    Privacy Policy
                  </Link>
                  {" "}and{" "}
                  <Link href="/cookies" className="text-blue-400 hover:underline">
                    Cookie Policy
                  </Link>
                  .
                </p>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// Function to open cookie settings from anywhere in the app
export function openCookieSettings() {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("openCookieSettings"));
  }
}

// Hook to check if analytics consent was granted
export function useAnalyticsConsent(): boolean {
  const [hasConsent, setHasConsent] = useState(false);

  useEffect(() => {
    const checkConsent = () => {
      const preferences = localStorage.getItem(COOKIE_PREFERENCES_KEY);
      if (preferences) {
        const parsed = JSON.parse(preferences) as CookiePreferences;
        setHasConsent(parsed.analytics);
      }
    };

    checkConsent();

    // Listen for consent changes
    const handleConsentChange = () => checkConsent();
    window.addEventListener("cookieConsentGranted", handleConsentChange);
    window.addEventListener("storage", handleConsentChange);

    return () => {
      window.removeEventListener("cookieConsentGranted", handleConsentChange);
      window.removeEventListener("storage", handleConsentChange);
    };
  }, []);

  return hasConsent;
}
