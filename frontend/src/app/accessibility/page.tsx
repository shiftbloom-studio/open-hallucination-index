import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Accessibility Statement",
};

export default function AccessibilityPage() {
  return (
    <main className="min-h-screen bg-black/[0.96] text-neutral-100 relative overflow-hidden">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <h1 className="text-4xl font-bold mb-6">Accessibility Statement</h1>

        <div className="space-y-6 text-neutral-200">
          <p className="text-sm text-neutral-400">Last Updated: January 6, 2026</p>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">Our Commitment to Accessibility</h2>
            <p>
              shiftbloom studio. is committed to ensuring digital accessibility for people with disabilities. We are continually improving the user experience for everyone and applying the relevant accessibility standards to ensure we provide equal access to all of our users.
            </p>
            <p>
              We believe that the internet should be available and accessible to anyone, and we are dedicated to providing a website and services that are accessible to the widest possible audience, regardless of technology or ability.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">Conformance Status</h2>
            <p>
              The Web Content Accessibility Guidelines (WCAG) define requirements for designers and developers to improve accessibility for people with disabilities. We aim to conform to WCAG 2.1 Level AA standards.
            </p>
            <p>
              Current status: <strong>Partially Conformant</strong>
            </p>
            <p>
              &quot;Partially conformant&quot; means that some parts of the content do not fully conform to the accessibility standard. We are actively working to achieve full conformance.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">Accessibility Features</h2>
            <p>
              We have implemented the following accessibility features on our website:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li><strong>Keyboard Navigation:</strong> All functionality is accessible using only a keyboard</li>
              <li><strong>Screen Reader Compatibility:</strong> Our site is designed to work with popular screen readers</li>
              <li><strong>Alternative Text:</strong> Images include descriptive alternative text</li>
              <li><strong>Color Contrast:</strong> We maintain sufficient color contrast ratios for text readability</li>
              <li><strong>Resizable Text:</strong> Text can be resized up to 200% without loss of content or functionality</li>
              <li><strong>Focus Indicators:</strong> Clear visual indicators show which element has keyboard focus</li>
              <li><strong>Semantic HTML:</strong> We use proper HTML markup to convey structure and meaning</li>
              <li><strong>Skip Links:</strong> Skip navigation links allow users to bypass repetitive content</li>
              <li><strong>Form Labels:</strong> All form fields have associated labels for screen reader users</li>
              <li><strong>Error Identification:</strong> Form errors are clearly identified and described</li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">Technologies Used</h2>
            <p>
              Accessibility of Open Hallucination Index relies on the following technologies to work with the particular combination of web browser and any assistive technologies or plugins installed on your computer:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>HTML5</li>
              <li>CSS3</li>
              <li>JavaScript</li>
              <li>WAI-ARIA (Web Accessibility Initiative – Accessible Rich Internet Applications)</li>
            </ul>
            <p>
              These technologies are relied upon for conformance with the accessibility standards used.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">Known Limitations</h2>
            <p>
              Despite our best efforts, there may be some limitations. Below is a description of known limitations and potential solutions:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>
                <strong>Interactive Visualizations:</strong> Some data visualizations and 3D elements may not be fully accessible to screen readers. We provide alternative text descriptions where possible.
              </li>
              <li>
                <strong>Third-Party Content:</strong> Some third-party content (e.g., payment processing via Stripe) may have accessibility limitations outside our control.
              </li>
              <li>
                <strong>PDF Documents:</strong> Some older PDF documents may not be fully accessible. We are working to remediate these.
              </li>
              <li>
                <strong>Dynamic Content:</strong> Some dynamically loaded content may not be immediately announced by screen readers. We are implementing ARIA live regions to address this.
              </li>
            </ul>
            <p>
              We are actively working to address these limitations and improve accessibility across our platform.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">Assistive Technologies Supported</h2>
            <p>
              Our website is designed to be compatible with the following assistive technologies:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Screen readers (JAWS, NVDA, VoiceOver, TalkBack)</li>
              <li>Screen magnification software</li>
              <li>Speech recognition software</li>
              <li>Alternative input devices</li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">Browser Compatibility</h2>
            <p>
              For the best accessible experience, we recommend using the latest versions of:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Google Chrome</li>
              <li>Mozilla Firefox</li>
              <li>Apple Safari</li>
              <li>Microsoft Edge</li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">Feedback and Contact</h2>
            <p>
              We welcome your feedback on the accessibility of Open Hallucination Index. If you encounter any accessibility barriers or have suggestions for improvement, please contact us:
            </p>
            <p>
              <strong>Email:</strong> hi@shiftbloom.studio<br />
              <strong>Subject Line:</strong> Accessibility Feedback
            </p>
            <p>
              When contacting us, please include:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>The web address (URL) of the page where you encountered the issue</li>
              <li>A description of the accessibility problem</li>
              <li>The browser and assistive technology you were using</li>
              <li>Any other relevant details that may help us understand and address the issue</li>
            </ul>
            <p>
              We aim to respond to accessibility feedback within 5 business days.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">Enforcement Procedure</h2>
            <p>
              If you are not satisfied with our response to your accessibility concern, you may escalate your complaint to the appropriate regulatory authority in your jurisdiction.
            </p>
            <p>
              For users in Germany, you may contact the Schlichtungsstelle nach dem Behindertengleichstellungsgesetz (BGG):
            </p>
            <p>
              Schlichtungsstelle BGG<br />
              bei dem Beauftragten der Bundesregierung für die Belange von Menschen mit Behinderungen<br />
              Mauerstraße 53<br />
              10117 Berlin<br />
              Germany<br />
              Email: info@schlichtungsstelle-bgg.de
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">Assessment and Evaluation</h2>
            <p>
              This website was assessed using the following approaches:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Self-evaluation</li>
              <li>Automated testing tools (Lighthouse, axe DevTools, WAVE)</li>
              <li>Manual testing with assistive technologies</li>
              <li>Keyboard-only navigation testing</li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">Continuous Improvement</h2>
            <p>
              We view accessibility as an ongoing effort rather than a one-time project. Our approach includes:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Regular accessibility audits and testing</li>
              <li>Training our team on accessibility best practices</li>
              <li>Including accessibility requirements in our development process</li>
              <li>Monitoring and addressing user feedback</li>
              <li>Staying updated on accessibility guidelines and standards</li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">Legal Framework</h2>
            <p>
              This accessibility statement is prepared in accordance with:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>EU Directive 2016/2102 (Web Accessibility Directive)</li>
              <li>German BITV 2.0 (Barrierefreie-Informationstechnik-Verordnung)</li>
              <li>WCAG 2.1 Guidelines (Web Content Accessibility Guidelines)</li>
            </ul>
          </section>

          <hr className="border-white/10 my-8" />

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">Contact Information</h2>
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
            This Accessibility Statement was last reviewed on January 6, 2026.
          </p>
        </div>
      </div>
    </main>
  );
}
