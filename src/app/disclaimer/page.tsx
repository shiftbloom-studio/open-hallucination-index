import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Liability Disclaimer",
};

export default function DisclaimerPage() {
  return (
    <main className="min-h-screen bg-black/[0.96] text-neutral-100 relative overflow-hidden">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <h1 className="text-4xl font-bold mb-6">Liability Disclaimer</h1>

        <div className="space-y-6 text-neutral-200">
          <p className="text-sm text-neutral-400">Last Updated: January 6, 2026</p>
          <p className="text-sm text-neutral-400">Effective Date: January 6, 2026</p>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">1. General Information</h2>
            <p>
              The information and services provided on this website (&quot;Open Hallucination Index&quot;) and through our platform are provided &quot;as is&quot; and &quot;as available&quot; without any warranties of any kind, either express or implied. This disclaimer applies to all content, features, and services offered by shiftbloom studio. (&quot;we,&quot; &quot;us,&quot; or &quot;our&quot;).
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">2. No Professional Advice</h2>
            <p>
              The content provided through our Service, including but not limited to hallucination detection results, verification scores, and analysis outputs, is for informational purposes only. It does not constitute:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Legal advice or legal opinion</li>
              <li>Professional consulting services</li>
              <li>Scientific or academic peer review</li>
              <li>Medical, financial, or other professional advice</li>
              <li>Definitive statements of fact or truth</li>
            </ul>
            <p>
              You should not rely solely on our Service&apos;s outputs for any critical decisions. Always consult qualified professionals for matters requiring expert judgment.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">3. Accuracy and Completeness</h2>
            <p>
              While we strive to provide accurate and up-to-date information, we make no representations or warranties regarding:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>The accuracy, reliability, or completeness of any analysis results</li>
              <li>The correctness of hallucination detection or verification scores</li>
              <li>The suitability of our Service for any particular purpose</li>
              <li>The continuous availability or error-free operation of the Service</li>
              <li>The timeliness or currency of information provided</li>
            </ul>
            <p>
              AI-based systems, including ours, may produce incorrect, incomplete, or misleading results. Our verification system is a tool to assist human judgment, not to replace it.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">4. Limitation of Liability</h2>
            <p>
              <strong>4.1 General Limitation.</strong> To the maximum extent permitted by applicable law, shiftbloom studio., its officers, directors, employees, agents, and affiliates shall not be liable for:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Any direct, indirect, incidental, special, consequential, or punitive damages</li>
              <li>Loss of profits, revenue, data, use, goodwill, or other intangible losses</li>
              <li>Damages resulting from unauthorized access to or use of our servers</li>
              <li>Damages resulting from any interruption or cessation of transmission to or from the Service</li>
              <li>Damages resulting from bugs, viruses, trojan horses, or similar harmful code</li>
              <li>Damages resulting from errors or omissions in any content or analysis</li>
              <li>Damages arising from decisions made based on Service outputs</li>
            </ul>
            <p>
              <strong>4.2 Exceptions.</strong> This limitation does not apply to liability for:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Death or personal injury caused by our negligence</li>
              <li>Fraud or fraudulent misrepresentation</li>
              <li>Any liability that cannot be excluded or limited under applicable law</li>
              <li>Intentional misconduct or gross negligence</li>
            </ul>
            <p>
              <strong>4.3 Liability Cap.</strong> Where liability cannot be fully excluded, our total aggregate liability shall not exceed the greater of (a) the amount you paid to us in the twelve (12) months preceding the claim, or (b) one hundred euros (€100).
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">5. AI and Machine Learning Disclaimer</h2>
            <p>
              Our Service utilizes artificial intelligence and machine learning technologies. By using our Service, you acknowledge and agree that:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>AI systems may produce unexpected, incorrect, or biased results</li>
              <li>The accuracy of AI outputs can vary based on input quality, context, and other factors</li>
              <li>AI systems learn and evolve, which may cause changes in output quality over time</li>
              <li>No AI system can guarantee 100% accuracy or reliability</li>
              <li>AI-generated content should be reviewed and verified by qualified humans</li>
              <li>We do not guarantee that our AI will detect all hallucinations or inaccuracies</li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">6. Third-Party Content and Links</h2>
            <p>
              Our Service may contain links to third-party websites, services, or content. We do not:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Control, endorse, or assume responsibility for third-party content</li>
              <li>Guarantee the accuracy, safety, or legality of third-party materials</li>
              <li>Accept liability for any loss or damage arising from third-party interactions</li>
            </ul>
            <p>
              Your use of third-party services is at your own risk and subject to the terms and conditions of those third parties.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">7. User Responsibility</h2>
            <p>
              You are solely responsible for:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Evaluating the accuracy and usefulness of any information or content obtained through our Service</li>
              <li>Any actions taken or decisions made based on information from our Service</li>
              <li>Ensuring your use of the Service complies with applicable laws and regulations</li>
              <li>Maintaining appropriate backups of your data</li>
              <li>Implementing adequate security measures for your account</li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">8. Service Availability</h2>
            <p>
              We do not guarantee that our Service will be:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Available at all times or without interruption</li>
              <li>Free from errors, bugs, or security vulnerabilities</li>
              <li>Compatible with all devices, browsers, or systems</li>
              <li>Maintained indefinitely or in its current form</li>
            </ul>
            <p>
              We reserve the right to modify, suspend, or discontinue any part of the Service at any time without prior notice.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">9. Indemnification</h2>
            <p>
              You agree to indemnify and hold harmless shiftbloom studio. and its officers, directors, employees, and agents from any claims, damages, losses, liabilities, and expenses (including reasonable legal fees) arising from:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Your use of the Service</li>
              <li>Your violation of this Disclaimer or any applicable law</li>
              <li>Your infringement of any third-party rights</li>
              <li>Any content you submit through the Service</li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">10. Governing Law</h2>
            <p>
              This Disclaimer shall be governed by and construed in accordance with the laws of the Federal Republic of Germany, without regard to its conflict of law provisions. If you are a consumer in the European Union, you may also benefit from mandatory provisions of the consumer protection laws of your country of residence.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">11. Severability</h2>
            <p>
              If any provision of this Disclaimer is found to be invalid or unenforceable, the remaining provisions shall continue in full force and effect. The invalid or unenforceable provision shall be replaced with a valid provision that most closely reflects the intent of the original provision.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">12. Changes to This Disclaimer</h2>
            <p>
              We reserve the right to update or modify this Disclaimer at any time. Changes will be effective immediately upon posting. Your continued use of the Service after any changes constitutes acceptance of the modified Disclaimer.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">13. Contact Information</h2>
            <p>
              If you have any questions about this Liability Disclaimer, please contact us at:
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

          <hr className="border-white/10 my-8" />
          <p className="text-xs text-neutral-500">
            Note: This Liability Disclaimer was created as a template and does not replace individual legal advice. Please review with qualified legal counsel to ensure compliance with applicable laws.
          </p>
        </div>
      </div>
    </main>
  );
}
