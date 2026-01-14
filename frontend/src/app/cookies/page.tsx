import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Cookie Policy",
};

export default function CookiePolicyPage() {
  return (
    <main className="min-h-screen bg-black/[0.96] text-neutral-100 relative overflow-hidden">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <h1 className="text-4xl font-bold mb-6">Cookie Policy</h1>

        <div className="space-y-6 text-neutral-200">
          <p className="text-sm text-neutral-400">Last Updated: January 6, 2026</p>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">1. What Are Cookies?</h2>
            <p>
              Cookies are small text files that are placed on your computer or mobile device when you visit a website. They are widely used to make websites work more efficiently, provide information to website owners, and enable certain features.
            </p>
            <p>
              Cookies may be &quot;persistent&quot; cookies or &quot;session&quot; cookies. Persistent cookies remain on your device after you close your browser, while session cookies are deleted when you close your browser.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">2. How We Use Cookies</h2>
            <p>
              We use cookies and similar technologies on our website Open Hallucination Index for the following purposes:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li><strong>Essential functionality:</strong> To provide core features like secure login, session management, and access to protected areas</li>
              <li><strong>Preferences:</strong> To remember your choices and settings</li>
              <li><strong>Analytics:</strong> To understand how visitors use our website and improve our services</li>
              <li><strong>Security:</strong> To protect against fraud and unauthorized access</li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">3. Types of Cookies We Use</h2>
            
            <div className="bg-slate-800/50 rounded-lg p-4 space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-neutral-100 mb-2">Strictly Necessary Cookies</h3>
                <p className="text-sm text-slate-300 mb-2">
                  These cookies are essential for the website to function properly and cannot be disabled.
                </p>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left border-b border-slate-700">
                      <th className="py-2 text-slate-400">Cookie Name</th>
                      <th className="py-2 text-slate-400">Purpose</th>
                      <th className="py-2 text-slate-400">Duration</th>
                    </tr>
                  </thead>
                  <tbody className="text-slate-300">
                    <tr className="border-b border-slate-700/50">
                      <td className="py-2 font-mono text-xs">sb-*-auth-token</td>
                      <td className="py-2">Supabase authentication session</td>
                      <td className="py-2">Session</td>
                    </tr>
                    <tr className="border-b border-slate-700/50">
                      <td className="py-2 font-mono text-xs">ohi_cookie_consent</td>
                      <td className="py-2">Stores your cookie consent choice</td>
                      <td className="py-2">1 year</td>
                    </tr>
                    <tr>
                      <td className="py-2 font-mono text-xs">ohi_cookie_preferences</td>
                      <td className="py-2">Stores your cookie preferences</td>
                      <td className="py-2">1 year</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-neutral-100 mb-2">Analytics Cookies</h3>
                <p className="text-sm text-slate-300 mb-2">
                  These cookies help us understand how visitors interact with our website. They are only set if you consent.
                </p>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left border-b border-slate-700">
                      <th className="py-2 text-slate-400">Cookie Name</th>
                      <th className="py-2 text-slate-400">Purpose</th>
                      <th className="py-2 text-slate-400">Duration</th>
                    </tr>
                  </thead>
                  <tbody className="text-slate-300">
                    <tr>
                      <td className="py-2 font-mono text-xs">va_*</td>
                      <td className="py-2">Vercel Analytics - anonymous usage statistics</td>
                      <td className="py-2">Session</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-neutral-100 mb-2">Functional Cookies</h3>
                <p className="text-sm text-slate-300 mb-2">
                  These cookies enable enhanced functionality and personalization. They are only set if you consent.
                </p>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left border-b border-slate-700">
                      <th className="py-2 text-slate-400">Cookie Name</th>
                      <th className="py-2 text-slate-400">Purpose</th>
                      <th className="py-2 text-slate-400">Duration</th>
                    </tr>
                  </thead>
                  <tbody className="text-slate-300">
                    <tr>
                      <td className="py-2 font-mono text-xs">ohi_preferences</td>
                      <td className="py-2">Stores user interface preferences</td>
                      <td className="py-2">1 year</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">4. Third-Party Cookies</h2>
            <p>
              We may use third-party services that set their own cookies. These include:
            </p>
            
            <div className="space-y-4">
              <div className="bg-slate-800/50 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-neutral-100 mb-2">Vercel Analytics</h3>
                <p className="text-sm text-slate-300">
                  We use Vercel Analytics to collect anonymous, aggregated data about how visitors use our website. Vercel Analytics is privacy-focused and does not collect personal data. It helps us understand page views, user flows, and website performance.
                </p>
                <p className="text-sm text-slate-400 mt-2">
                  Privacy Policy:{" "}
                  <a href="https://vercel.com/legal/privacy-policy" className="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">
                    https://vercel.com/legal/privacy-policy
                  </a>
                </p>
              </div>

              <div className="bg-slate-800/50 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-neutral-100 mb-2">Supabase</h3>
                <p className="text-sm text-slate-300">
                  We use Supabase for user authentication. Supabase sets cookies necessary for secure login and session management.
                </p>
                <p className="text-sm text-slate-400 mt-2">
                  Privacy Policy:{" "}
                  <a href="https://supabase.com/privacy" className="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">
                    https://supabase.com/privacy
                  </a>
                </p>
              </div>

              <div className="bg-slate-800/50 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-neutral-100 mb-2">Stripe</h3>
                <p className="text-sm text-slate-300">
                  We use Stripe for payment processing. When you make a purchase, Stripe may set cookies for fraud prevention and payment security. These cookies are set only during the checkout process on Stripe&apos;s domain.
                </p>
                <p className="text-sm text-slate-400 mt-2">
                  Privacy Policy:{" "}
                  <a href="https://stripe.com/privacy" className="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">
                    https://stripe.com/privacy
                  </a>
                </p>
              </div>
            </div>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">5. Managing Cookies</h2>
            <p>
              You can manage your cookie preferences in several ways:
            </p>
            <ul className="list-disc pl-6 space-y-2">
              <li>
                <strong>Cookie Banner:</strong> When you first visit our website, you can choose which categories of cookies to accept using our cookie consent banner.
              </li>
              <li>
                <strong>Browser Settings:</strong> Most web browsers allow you to control cookies through their settings. You can set your browser to refuse cookies or delete specific cookies. However, if you block all cookies, some features of our website may not work properly.
              </li>
            </ul>
            
            <p className="mt-4">
              Here&apos;s how to manage cookies in popular browsers:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>
                <a href="https://support.google.com/chrome/answer/95647" className="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">
                  Google Chrome
                </a>
              </li>
              <li>
                <a href="https://support.mozilla.org/en-US/kb/cookies-information-websites-store-on-your-computer" className="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">
                  Mozilla Firefox
                </a>
              </li>
              <li>
                <a href="https://support.apple.com/guide/safari/manage-cookies-sfri11471/mac" className="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">
                  Safari
                </a>
              </li>
              <li>
                <a href="https://support.microsoft.com/en-us/windows/manage-cookies-in-microsoft-edge-view-allow-block-delete-and-use-168dab11-0753-043d-7c16-ede5947fc64d" className="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">
                  Microsoft Edge
                </a>
              </li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">6. Legal Basis for Processing</h2>
            <p>
              Under the General Data Protection Regulation (GDPR), we process cookies based on the following legal bases:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>
                <strong>Strictly necessary cookies:</strong> Art. 6(1)(f) GDPR (legitimate interest in providing a functional website)
              </li>
              <li>
                <strong>Analytics and functional cookies:</strong> Art. 6(1)(a) GDPR (your consent)
              </li>
            </ul>
            <p className="mt-2">
              You may withdraw your consent at any time by clearing your cookies and revisiting our website, or by adjusting your browser settings.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">7. Updates to This Policy</h2>
            <p>
              We may update this Cookie Policy from time to time. When we make significant changes, we will notify you by updating the &quot;Last Updated&quot; date at the top of this policy and, where appropriate, provide additional notice.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">8. Contact Us</h2>
            <p>
              If you have any questions about our use of cookies, please contact us at:
            </p>
            <p>
              shiftbloom studio.<br />
              Fabian Zimber<br />
              Up de Worth 6a<br />
              22927 Großhansdorf<br />
              Germany<br />
              Email: hi@shiftbloom.studio
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">9. Related Policies</h2>
            <p>
              For more information about how we handle your personal data, please refer to our:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>
                <Link href="/datenschutz" className="text-blue-400 hover:underline">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link href="/agb" className="text-blue-400 hover:underline">
                  Terms of Service
                </Link>
              </li>
            </ul>
          </section>

          <hr className="border-white/10 my-8" />
          <p className="text-xs text-neutral-500">
            This Cookie Policy complies with the EU General Data Protection Regulation (GDPR), the ePrivacy Directive, and the German Telemediengesetz (TMG) / Telekommunikation-Telemedien-Datenschutz-Gesetz (TTDSG).
          </p>
        </div>
      </div>
    </main>
  );
}
