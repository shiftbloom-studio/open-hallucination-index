import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Terms of Service",
};

export default function TermsOfServicePage() {
  return (
    <main className="sb-legal-page">
      <div className="mx-auto max-w-4xl px-6 py-32 md:py-40">
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
              The prices displayed in the checkout at the time of the order are final and include any applicable Value Added Tax (VAT) as required by law. Prices are stated in Euro (€).
            </p>
            <p>
              For customers within the European Union, VAT is calculated based on the applicable rate in your country of residence. For non-EU customers, VAT may not apply.
            </p>
            <p>
              Payment is due immediately upon order completion. Accepted payment methods are displayed in the checkout and are processed through our secure payment provider, Stripe.
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
              <strong>Important Notice:</strong> Consumers in the European Union are generally entitled to a statutory right of withdrawal for distance contracts. However, this right expires for digital content under specific conditions as outlined below.
            </p>
            <p>
              <strong>Expiration of the Right of Withdrawal:</strong> The right of withdrawal expires if you have expressly consented to us beginning performance of the contract before the expiration of the withdrawal period and have acknowledged that you lose your right of withdrawal once performance has begun.
            </p>
            <p>
              By purchasing OHI Tokens, you expressly consent to immediate delivery of the digital content (token crediting to your account) and acknowledge that you thereby lose your right of withdrawal once the tokens are credited to your account.
            </p>
            <p>
              <strong>No Money-Back Guarantee:</strong> We do not offer refunds or money-back guarantees for purchased tokens after they have been credited to your account. All sales are final once the digital content has been delivered. Please ensure you understand the Service and are satisfied with your purchase before completing checkout.
            </p>
            <p>
              If you experience technical issues with token delivery or service functionality, please contact us at hi@shiftbloom.studio, and we will work to resolve the issue in accordance with our warranty obligations.
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
              User accounts can be terminated by users at any time by contacting us via email at hi@shiftbloom.studio or through the account settings interface.
            </p>
            <p>
              <strong>Token Expiration Policy:</strong> Purchased OHI Tokens do not expire and remain available in your account indefinitely, even after account termination, unless the account is permanently deleted at your request. If you request permanent account deletion, all unused tokens will be forfeited without refund.
            </p>
            <p>
              The right to extraordinary termination for good cause remains unaffected. We reserve the right to terminate accounts that violate these Terms, engage in fraudulent activity, or pose security risks.
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
            These Terms of Service were last updated on January 6, 2026. We reserve the right to modify these terms at any time. Material changes will be communicated to registered users via email or through prominent notice on the Service.
          </p>
        </div>
      </div>
    </main>
  );
}
