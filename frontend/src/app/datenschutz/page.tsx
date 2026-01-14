import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy",
};

export default function PrivacyPolicyPage() {
  return (
    <main className="min-h-screen bg-black/[0.96] text-neutral-100 relative overflow-hidden">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <h1 className="text-4xl font-bold mb-6">Privacy Policy</h1>

        <div className="space-y-6 text-neutral-200">
          <p className="text-sm text-neutral-400">Last Updated: January 6, 2026</p>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">1. Data Controller</h2>
            <p>
              The data controller within the meaning of the General Data Protection Regulation (GDPR) is:
              <br />
              shiftbloom studio.<br />
              Fabian Zimber<br />
              Up de Worth 6a, 22927 Großhansdorf, Germany<br />
              Email: hi@shiftbloom.studio
            </p>
            <p>
              (Note: If a Data Protection Officer has been appointed, please add the contact details.)
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">2. Overview: What Data Do We Process?</h2>
            <p>
              Depending on the use of the Service, we process in particular:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Inventory data (e.g., email, account ID)</li>
              <li>Usage/metadata (e.g., login status, token balance, technically required identifiers)</li>
              <li>Content data that you enter/upload in the Service (e.g., texts for analysis)</li>
              <li>Payment/transaction data (in connection with Stripe checkout)</li>
              <li>Log data (e.g., IP address, timestamp, request information in server logs)</li>
              <li>Communication data (e.g., email content for support inquiries)</li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">3. Purposes and Legal Bases</h2>
            <p>
              We process personal data only to the extent permitted. Typical purposes and legal bases are:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>
                Contract performance and pre-contractual measures (Art. 6(1)(b) GDPR), e.g., registration, login, provision of the Service, token crediting.
              </li>
              <li>
                Legal obligations (Art. 6(1)(c) GDPR), e.g., commercial/tax law retention requirements.
              </li>
              <li>
                Legitimate interests (Art. 6(1)(f) GDPR), e.g., IT security, fraud prevention, error analysis, operation and optimization.
              </li>
              <li>
                Consent (Art. 6(1)(a) GDPR), for analytics cookies and optional tracking. We use Vercel Analytics with your consent to understand how visitors use our website.
              </li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">4. Hosting (Vercel) and Server Logs</h2>
            <p>
              Our website is hosted on Vercel Inc., 440 N Barranca Ave #4133, Covina, CA 91723, USA. When visiting the website, Vercel processes technically necessary data and stores it in log files (e.g., IP address, date/time, page accessed, referrer URL, user agent, status codes). Processing occurs for website delivery, IT security, and error analysis.
            </p>
            <p>
              Legal basis is Art. 6(1)(f) GDPR (legitimate interest in secure and stable operation). Vercel may transfer data to the USA. Transfer occurs under appropriate safeguards (e.g., Standard Contractual Clauses and the EU-U.S. Data Privacy Framework). For details, please refer to Vercel&apos;s privacy policy at{" "}
              <a href="https://vercel.com/legal/privacy-policy" className="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">
                https://vercel.com/legal/privacy-policy
              </a>.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">5. Authentication and User Account (Supabase)</h2>
            <p>
              For registration/login, we use Supabase. This involves processing your email address, technical identifiers, and session information. To maintain your login, Supabase sets technically necessary cookies/tokens (session cookies).
            </p>
            <p>
              Legal basis is Art. 6(1)(b) GDPR (contract performance) and Art. 6(1)(f) GDPR (security and fraud prevention).
            </p>
            <p>
              Recipients: Supabase (depending on configuration, potentially in third countries). If transfer to a third country occurs, it is based – where required – on appropriate safeguards (e.g., Standard Contractual Clauses). For details, please refer to Supabase&apos;s privacy notices.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">6. Payment Processing (Stripe)</h2>
            <p>
              For the purchase of token packages, we use Stripe as payment service provider. When you make a purchase, you will be redirected to Stripe checkout. Stripe processes payment data (e.g., card details) that we do not fully access.
            </p>
            <p>
              We transmit to Stripe in particular:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Email address (for assignment/receipt communication)</li>
              <li>Technical identifiers (e.g., session/transaction data)</li>
              <li>Metadata for assignment in the Service (e.g., user ID, package ID)</li>
            </ul>
            <p>
              Legal basis is Art. 6(1)(b) GDPR (payment processing/contract performance) and, where applicable, Art. 6(1)(c) GDPR (retention requirements).
            </p>
            <p>
              Stripe may transfer data to third countries (e.g., USA). Where required, transfer occurs under appropriate safeguards (e.g., Standard Contractual Clauses). For details, please refer to Stripe&apos;s privacy notices.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">7. Database / Storage</h2>
            <p>
              To provide the Service, we store data in a database (e.g., user account, token balance, settings). Storage occurs as long as required for contract performance or as long as statutory retention obligations exist.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">8. Cookies</h2>
            <p>
              We use cookies to the extent they are technically necessary (e.g., for login/session). Technically necessary cookies are required for the Service to function.
            </p>
            <p>
              For analytics cookies (Vercel Analytics), we obtain your consent before setting them. You can manage your cookie preferences at any time using our cookie consent banner. For detailed information about the cookies we use, please see our{" "}
              <a href="/cookies" className="text-blue-400 hover:underline">
                Cookie Policy
              </a>.
            </p>
            <p>
              Legal basis for necessary cookies is Art. 6(1)(f) GDPR (legitimate interest). For optional/analytics cookies, the legal basis is Art. 6(1)(a) GDPR (consent).
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">8a. Web Analytics (Vercel Analytics)</h2>
            <p>
              With your consent, we use Vercel Analytics to analyze website usage. Vercel Analytics is a privacy-focused analytics service that collects anonymous, aggregated data about page views, user flows, and website performance.
            </p>
            <p>
              <strong>Data collected:</strong> Page URL, referrer, browser type, device type, country (derived from IP, but IP is not stored), screen size, and interaction events. No personal data such as IP addresses or user identifiers are stored.
            </p>
            <p>
              <strong>Purpose:</strong> Understanding how visitors use our website, improving user experience, and optimizing our services.
            </p>
            <p>
              <strong>Legal basis:</strong> Art. 6(1)(a) GDPR (your consent). You can withdraw your consent at any time by adjusting your cookie preferences.
            </p>
            <p>
              <strong>Recipient:</strong> Vercel Inc., USA. Data transfer to the USA occurs under appropriate safeguards (EU-U.S. Data Privacy Framework, Standard Contractual Clauses).
            </p>
            <p>
              For more information, see Vercel&apos;s privacy policy at{" "}
              <a href="https://vercel.com/legal/privacy-policy" className="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">
                https://vercel.com/legal/privacy-policy
              </a>.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">9. Contact</h2>
            <p>
              When you contact us by email, we process your information to handle the inquiry. Legal basis is Art. 6(1)(b) GDPR (initiation/performance) or Art. 6(1)(f) GDPR (legitimate interest in efficient communication).
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">10. Recipients, Processors</h2>
            <p>
              We use service providers who process personal data on our behalf. With these service providers, we conclude – where required – data processing agreements (Art. 28 GDPR).
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li><strong>Vercel Inc.</strong> (USA) – Hosting and analytics</li>
              <li><strong>Supabase Inc.</strong> – Authentication and database</li>
              <li><strong>Stripe Inc.</strong> (USA) – Payment processing</li>
            </ul>
            <p>
              For transfers to third countries (e.g., USA), we rely on appropriate safeguards such as Standard Contractual Clauses and the EU-U.S. Data Privacy Framework.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">11. Retention Period</h2>
            <p>
              We store personal data only as long as required for the respective purposes. Beyond this, we store data to the extent statutory retention obligations exist (e.g., tax/commercial law).
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">12. Your Rights</h2>
            <p>
              Depending on legal requirements, you are entitled to the following rights:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Access (Art. 15 GDPR)</li>
              <li>Rectification (Art. 16 GDPR)</li>
              <li>Erasure (Art. 17 GDPR)</li>
              <li>Restriction of processing (Art. 18 GDPR)</li>
              <li>Data portability (Art. 20 GDPR)</li>
              <li>Objection to processing (Art. 21 GDPR)</li>
              <li>Withdrawal of consent (Art. 7(3) GDPR) with effect for the future</li>
              <li>Complaint to a supervisory authority (Art. 77 GDPR)</li>
            </ul>
            <p>
              To exercise your rights, a message to hi@shiftbloom.studio is sufficient.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">13. Obligation to Provide Data</h2>
            <p>
              For registration and use of the Service, the provision of certain data (e.g., email) is required. Without this data, the Service cannot be used or can only be used to a limited extent.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">14. Changes to This Privacy Policy</h2>
            <p>
              We may update this Privacy Policy if legal requirements, services, or data processing change. The current version applies.
            </p>
          </section>

          <hr className="border-white/10 my-8" />
          <p className="text-xs text-neutral-500">
            This Privacy Policy complies with the EU General Data Protection Regulation (GDPR), the German BDSG, and the TTDSG. Note: This document serves as a template and does not replace individual legal advice.
          </p>
        </div>
      </div>
    </main>
  );
}
