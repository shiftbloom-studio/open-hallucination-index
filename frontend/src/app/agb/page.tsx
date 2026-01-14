import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Terms of Service",
};

export default function TermsOfServicePage() {
  return (
    <main className="min-h-screen bg-black/[0.96] text-neutral-100 relative overflow-hidden">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <h1 className="text-4xl font-bold mb-6">Terms of Service</h1>

        <div className="space-y-6 text-neutral-200">
          <p className="text-sm text-neutral-400">Last Updated: January 6, 2026</p>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">1. Scope of Application</h2>
            <p>
              These Terms of Service (hereinafter &quot;Terms&quot;) apply to all contracts regarding the use of the online platform/web application &quot;Open Hallucination Index&quot; (hereinafter &quot;Service&quot;), including the purchase of token packages (&quot;OHI Tokens&quot;) as digital usage/consumption units within the Service.
            </p>
            <p>
              Deviating, conflicting, or supplementary terms from customers shall only become part of the contract if we have expressly agreed to their validity in writing.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">2. Provider / Contracting Party</h2>
            <p>
              The contracting party and provider of the Service is:
              <br />
              shiftbloom studio.<br />
              Fabian Zimber<br />
              Up de Worth 6a<br />
              22927 Großhansdorf<br />
              Germany<br />
              Email: hi@shiftbloom.studio
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">3. Service Description</h2>
            <p>
              The Service provides features that allow registered users to verify/analyze content and manage results in their account. Certain features require the consumption of OHI Tokens.
            </p>
            <p>
              OHI Tokens are not legal currency and are not cryptocurrency. They serve exclusively for using features within the Service.
            </p>
            <p>
              We owe the provision of the Service within the current state of technology. A specific success (e.g., certain analysis results, &quot;error-free operation,&quot; or suitability for a specific purpose) is not owed.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">4. Registration and User Account</h2>
            <p>
              The creation of a user account is required to use certain features. Truthful information must be provided during registration. Access credentials must be kept confidential; sharing with third parties is prohibited.
            </p>
            <p>
              We reserve the right to suspend or delete user accounts if there is concrete evidence of misuse, security risks, or significant violations of these Terms.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">5. Contract Conclusion (Token Purchase)</h2>
            <p>
              The display of token packages in the Service does not constitute a legally binding offer but an invitation to place an order.
            </p>
            <p>
              The contract for the acquisition of a token package is concluded when the payment process is completed and the payment confirmation/checkout is successful.
            </p>
            <p>
              Payment processing is handled by an external payment service provider (currently: Stripe). The terms of the payment service provider also apply.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">6. Prices, Taxes, Payment</h2>
            <p>
              The prices displayed in the checkout at the time of the order apply. If VAT is applicable and must be shown, it will be displayed accordingly in the checkout. (Note: Please verify tax details/status and adjust if necessary.)
            </p>
            <p>
              Payment is due immediately. Accepted payment methods are shown in the checkout.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">7. Delivery, Token Crediting</h2>
            <p>
              After successful payment receipt, the purchased OHI Tokens will be credited to the user account. Crediting typically occurs automatically; in exceptional cases, delays may occur.
            </p>
            <p>
              In case of technical problems, please contact us at hi@shiftbloom.studio with your order/payment information.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">8. Right of Withdrawal (Consumers)</h2>
            <p>
              Consumers are generally entitled to a statutory right of withdrawal. This right may expire for digital content under certain conditions.
            </p>
            <p>
              Expiration of the Right of Withdrawal: The right of withdrawal may expire if (1) we have begun performance of the contract after you (2) have expressly consented to us beginning performance before the expiration of the withdrawal period, and (3) you have confirmed your knowledge that by giving your consent with the beginning of performance you lose your right of withdrawal.
            </p>
            <p>
              Note: Whether and how this consent is technically obtained in the checkout should be reviewed and supplemented if necessary for legal compliance.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">9. Usage Rights and Permitted Use</h2>
            <p>
              To the extent the Service generates outputs/results, users receive a simple, non-exclusive, non-transferable right of use for their own purposes. Sharing, publication, or commercial exploitation may be restricted depending on the content/source.
            </p>
            <p>
              The following are specifically prohibited: (a) illegal content, (b) circumvention of security measures, (c) automated mass usage that impairs the Service, (d) reverse engineering, unless expressly permitted by law.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">10. Availability, Maintenance</h2>
            <p>
              We strive for high availability but do not guarantee uninterrupted availability. Maintenance, security updates, and technical disruptions may lead to temporary restrictions.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">11. Warranty</h2>
            <p>
              Statutory warranty rights apply. Digital services may have errors; we will process legitimate defect complaints in accordance with legal requirements.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">12. Liability</h2>
            <p>
              We are liable without limitation for intent and gross negligence, as well as for injury to life, body, or health.
            </p>
            <p>
              In case of simple negligence, we are only liable for breach of essential contractual obligations (cardinal obligations) and limited to the typical, foreseeable damage.
            </p>
            <p>
              Liability under the Product Liability Act remains unaffected.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">13. Term, Termination</h2>
            <p>
              User accounts can be terminated by users at any time (e.g., via email). Already purchased tokens may expire depending on the technical implementation; please establish specific rules accordingly. (Placeholder: please adapt to your token logic.)
            </p>
            <p>
              The right to extraordinary termination for good cause remains unaffected.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">14. Data Protection</h2>
            <p>
              Information about the processing of personal data can be found in our Privacy Policy.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">15. Final Provisions</h2>
            <p>
              The law of the Federal Republic of Germany applies, excluding the UN Convention on Contracts for the International Sale of Goods. If you are a consumer, this choice of law only applies to the extent that the protection afforded by mandatory provisions of the law of your country of habitual residence is not withdrawn.
            </p>
            <p>
              If you are a merchant, legal entity under public law, or special fund under public law, the place of jurisdiction – as far as permissible – is the provider&apos;s registered office.
            </p>
            <p>
              Should individual provisions of these Terms be wholly or partially invalid, the validity of the remaining provisions shall remain unaffected.
            </p>
          </section>

          <hr className="border-white/10 my-8" />
          <p className="text-xs text-neutral-500">
            Note: These Terms of Service were created as a template/draft and do not replace individual legal advice. Please adjust withdrawal/taxes/token expiration to your actual processes.
          </p>
        </div>
      </div>
    </main>
  );
}
