"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cookie, X, Settings, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { cn } from "@/lib/utils";

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
      ? { necessary: true, analytics: false, functional: true }
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
          <div className="sb-panel mx-auto max-w-4xl overflow-hidden bg-surface-elevated/95 backdrop-blur-xl">
            {!showSettings ? (
              // Main Banner
              <div className="p-6">
                <div className="flex items-start gap-4">
                  <div className="shrink-0 rounded-lg bg-[color:var(--brand-secondary)] p-3">
                    <Cookie className="h-6 w-6 text-[color:var(--brand-accent)]" />
                  </div>
                  <div className="flex-1 space-y-4">
                    <div>
                      <h3 className="mb-2 text-lg font-semibold text-brand-ink">
                        We Value Your Privacy
                      </h3>
                      <p className="text-sm leading-relaxed text-brand-muted">
                        We use necessary cookies for site preferences and may use optional functional cookies if you enable them. 
                        OHI does not currently set third-party analytics cookies. Your choices are described in our{" "}
                        <Link href="/datenschutz" className="text-[color:var(--brand-accent)] hover:underline">
                          Privacy Policy
                        </Link>
                        . You can customize your preferences or reject non-essential cookies.
                      </p>
                    </div>
                    
                    <div className="flex flex-wrap gap-3">
                      <Button
                        onClick={() => saveConsent(true)}
                        className="modern-btn min-h-0 px-6 py-2 text-sm"
                      >
                        <Check className="w-4 h-4 mr-2" />
                        Accept All
                      </Button>
                      <Button
                        onClick={rejectAll}
                        variant="outline"
                        className="tertiary-btn min-h-0 px-5 py-2 text-sm"
                      >
                        Reject All
                      </Button>
                      <Button
                        onClick={() => setShowSettings(true)}
                        variant="ghost"
                        className="text-brand-muted hover:bg-[color:var(--brand-secondary)] hover:text-brand-ink"
                      >
                        <Settings className="w-4 h-4 mr-2" />
                        Customize
                      </Button>
                    </div>
                  </div>
                  <button
                    onClick={rejectAll}
                    className="shrink-0 p-2 text-brand-muted transition-colors hover:text-brand-ink"
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
                  <h3 className="text-lg font-semibold text-brand-ink">Cookie Preferences</h3>
                  <button
                    onClick={() => setShowSettings(false)}
                    className="p-2 text-brand-muted transition-colors hover:text-brand-ink"
                    aria-label="Close settings"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                <div className="space-y-4 mb-6">
                  {/* Necessary Cookies */}
                  <div className="rounded-lg border border-[color:var(--border-subtle)] bg-surface-base p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <span className="font-medium text-brand-ink">Strictly Necessary</span>
                        <span className="rounded-full bg-[color:var(--brand-success-soft)] px-2 py-0.5 text-xs text-[color:var(--brand-success)]">
                          Always Active
                        </span>
                      </div>
                      <div className="flex h-6 w-12 cursor-not-allowed items-center justify-end rounded-full bg-[color:var(--brand-success)] px-1">
                        <div className="w-4 h-4 bg-white rounded-full shadow" />
                      </div>
                    </div>
                    <p className="text-sm text-brand-muted">
                      These cookies are essential for the website to function properly. They enable basic functions like page navigation, secure login, and access to secure areas. The website cannot function properly without these cookies.
                    </p>
                  </div>

                  {/* Analytics Cookies */}
                  <div className="rounded-lg border border-[color:var(--border-subtle)] bg-surface-base p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-brand-ink">Analytics & Performance</span>
                      <button
                        onClick={() => togglePreference("analytics")}
                        className={cn(
                          "flex h-6 w-12 items-center rounded-full px-1 transition-colors",
                          preferences.analytics
                            ? "justify-end bg-[color:var(--brand-success)]"
                            : "justify-start bg-[color:var(--border-default)]",
                        )}
                        role="switch"
                        aria-checked={preferences.analytics}
                        aria-label="Toggle analytics cookies"
                      >
                        <div className="w-4 h-4 bg-white rounded-full shadow" />
                      </button>
                    </div>
                    <p className="text-sm text-brand-muted">
                      OHI does not currently set third-party analytics cookies. This preference is reserved for optional analytics features if they are enabled later.
                    </p>
                  </div>

                  {/* Functional Cookies */}
                  <div className="rounded-lg border border-[color:var(--border-subtle)] bg-surface-base p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-brand-ink">Functional</span>
                      <button
                        onClick={() => togglePreference("functional")}
                        className={cn(
                          "flex h-6 w-12 items-center rounded-full px-1 transition-colors",
                          preferences.functional
                            ? "justify-end bg-[color:var(--brand-success)]"
                            : "justify-start bg-[color:var(--border-default)]",
                        )}
                        role="switch"
                        aria-checked={preferences.functional}
                        aria-label="Toggle functional cookies"
                      >
                        <div className="w-4 h-4 bg-white rounded-full shadow" />
                      </button>
                    </div>
                    <p className="text-sm text-brand-muted">
                      These cookies enable enhanced functionality and personalization, such as remembering your preferences, language settings, and providing personalized features. If you disable these, some features may not work as intended.
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-3 border-t border-[color:var(--border-subtle)] pt-4">
                  <Button
                    onClick={() => saveConsent(false)}
                    className="modern-btn min-h-0 px-6 py-2 text-sm"
                  >
                    Save Preferences
                  </Button>
                  <Button
                    onClick={() => saveConsent(true)}
                    variant="outline"
                    className="tertiary-btn min-h-0 px-5 py-2 text-sm"
                  >
                    Accept All
                  </Button>
                  <Button
                    onClick={rejectAll}
                    variant="ghost"
                    className="text-brand-muted hover:bg-[color:var(--brand-secondary)] hover:text-brand-ink"
                  >
                    Reject All
                  </Button>
                </div>

                <p className="mt-4 text-xs text-brand-subtle">
                  For more information, please read our{" "}
                  <Link href="/datenschutz" className="text-[color:var(--brand-accent)] hover:underline">
                    Privacy Policy
                  </Link>
                  {" "}and{" "}
                  <Link href="/cookies" className="text-[color:var(--brand-accent)] hover:underline">
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
