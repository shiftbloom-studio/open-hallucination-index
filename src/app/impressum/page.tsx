import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Legal Notice",
};

export default function LegalNoticePage() {
  return (
    <main className="min-h-screen bg-black/[0.96] text-neutral-100 relative overflow-hidden">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <h1 className="text-4xl font-bold mb-6">Impressum / Legal Notice</h1>

        <section className="space-y-4 text-neutral-200">
          <p className="text-sm text-neutral-400">Last Updated: January 6, 2026</p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Information pursuant to § 5 TMG (German Telemedia Act)</h2>
          <p>
            shiftbloom studio.<br />
            Fabian Zimber<br />
            Up de Worth 6a<br />
            22927 Großhansdorf<br />
            Germany
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Contact</h2>
          <p>
            Email: hi@shiftbloom.studio
            <br />
            Phone: (please add if desired)
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">VAT Information</h2>
          <p>
            VAT Identification Number pursuant to § 27a of the German VAT Act: (please add if applicable)
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Commercial Register Entry</h2>
          <p>
            Entry in the Commercial Register: (please add if applicable)<br />
            Register Court: (please add if applicable)<br />
            Registration Number: (please add if applicable)
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Responsible for Content pursuant to § 18 Sec. 2 MStV</h2>
          <p>
            Fabian Zimber, Up de Worth 6a, 22927 Großhansdorf, Germany
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">EU Online Dispute Resolution / Consumer Dispute Resolution</h2>
          <p>
            The European Commission provides a platform for online dispute resolution (ODR):
            <br />
            <a href="https://ec.europa.eu/consumers/odr/" className="text-blue-400 hover:underline">
              https://ec.europa.eu/consumers/odr/
            </a>
          </p>
          <p>
            We are neither obligated nor willing to participate in dispute resolution proceedings before a consumer arbitration board.
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Liability for Content</h2>
          <p>
            As a service provider, we are responsible for our own content on these pages in accordance with general laws pursuant to § 7 Sec. 1 TMG. However, according to §§ 8 to 10 TMG, we as a service provider are not obligated to monitor transmitted or stored third-party information or to investigate circumstances that indicate illegal activity.
          </p>
          <p>
            Obligations to remove or block the use of information under general law remain unaffected. However, liability in this regard is only possible from the point in time at which a specific legal violation becomes known. Upon becoming aware of any such legal violations, we will remove the content in question immediately.
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Liability for Links</h2>
          <p>
            Our website contains links to external third-party websites over whose content we have no influence. Therefore, we cannot accept any liability for this third-party content. The respective provider or operator of the linked pages is always responsible for the content of the linked pages.
          </p>
          <p>
            The linked pages were checked for possible legal violations at the time of linking. Illegal content was not recognizable at the time of linking. However, permanent monitoring of the content of the linked pages is not reasonable without concrete evidence of a legal violation. Upon becoming aware of any legal violations, we will remove such links immediately.
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Copyright</h2>
          <p>
            The content and works created by the site operators on these pages are subject to German copyright law. Reproduction, editing, distribution, and any kind of use outside the limits of copyright law require the written consent of the respective author or creator.
          </p>
          <p>
            Insofar as the content on this site was not created by the operator, the copyrights of third parties are respected. In particular, third-party content is marked as such. Should you nevertheless become aware of a copyright infringement, please notify us accordingly. Upon becoming aware of any legal violations, we will remove such content immediately.
          </p>

          <hr className="border-white/10 my-8" />
          <p className="text-xs text-neutral-500">
            Note: This legal notice was created as a template/draft and does not replace individual legal advice.
          </p>
        </section>
      </div>
    </main>
  );
}
