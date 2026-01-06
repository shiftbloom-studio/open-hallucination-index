"use client";

import { Analytics } from "@vercel/analytics/react";
import { useEffect, useState } from "react";

const COOKIE_PREFERENCES_KEY = "ohi_cookie_preferences";

interface CookiePreferences {
  necessary: boolean;
  analytics: boolean;
  functional: boolean;
}

export function ConsentAwareAnalytics() {
  const [hasConsent, setHasConsent] = useState(false);

  useEffect(() => {
    const checkConsent = () => {
      const preferences = localStorage.getItem(COOKIE_PREFERENCES_KEY);
      if (preferences) {
        try {
          const parsed = JSON.parse(preferences) as CookiePreferences;
          setHasConsent(parsed.analytics);
        } catch {
          setHasConsent(false);
        }
      }
    };

    // Initial check
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

  // Only render Analytics component if user has consented
  if (!hasConsent) {
    return null;
  }

  return <Analytics />;
}
