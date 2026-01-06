import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Impressum",
};

export default function ImpressumPage() {
  return (
    <main className="min-h-screen bg-black/[0.96] text-neutral-100 relative overflow-hidden">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <h1 className="text-4xl font-bold mb-6">Impressum</h1>

        <section className="space-y-4 text-neutral-200">
          <p className="text-sm text-neutral-400">Stand: 06.01.2026</p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Angaben gem. § 5 TMG</h2>
          <p>
            shiftbloom studio.<br />
            Fabian Zimber<br />
            Up de Worth 6a<br />
            22927 Großhansdorf<br />
            Deutschland
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Kontakt</h2>
          <p>
            E-Mail: hi@shiftbloom.studio
            <br />
            Telefon: (bitte ergänzen, falls gewünscht)
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Umsatzsteuer</h2>
          <p>
            Umsatzsteuer-Identifikationsnummer gemäß § 27 a Umsatzsteuergesetz: (falls vorhanden, bitte ergänzen)
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Registereintrag</h2>
          <p>
            Eintrag im Handelsregister: (falls vorhanden, bitte ergänzen)<br />
            Registergericht: (falls vorhanden, bitte ergänzen)<br />
            Registernummer: (falls vorhanden, bitte ergänzen)
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Verantwortlich für den Inhalt nach § 18 Abs. 2 MStV</h2>
          <p>
            Fabian Zimber, Up de Worth 6a, 22927 Großhansdorf, Deutschland
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Online-Streitbeilegung / Verbraucherstreitbeilegung</h2>
          <p>
            Die Europäische Kommission stellt eine Plattform zur Online-Streitbeilegung (OS) bereit:
            <br />
            https://ec.europa.eu/consumers/odr/
          </p>
          <p>
            Ich bin nicht verpflichtet und nicht bereit, an Streitbeilegungsverfahren vor einer Verbraucherschlichtungsstelle
            teilzunehmen.
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Haftung für Inhalte</h2>
          <p>
            Als Diensteanbieter bin ich gemäß § 7 Abs. 1 TMG für eigene Inhalte auf diesen Seiten nach den allgemeinen
            Gesetzen verantwortlich. Nach §§ 8 bis 10 TMG bin ich als Diensteanbieter jedoch nicht verpflichtet, übermittelte
            oder gespeicherte fremde Informationen zu überwachen oder nach Umständen zu forschen, die auf eine rechtswidrige
            Tätigkeit hinweisen.
          </p>
          <p>
            Verpflichtungen zur Entfernung oder Sperrung der Nutzung von Informationen nach den allgemeinen Gesetzen bleiben
            hiervon unberührt. Eine diesbezügliche Haftung ist jedoch erst ab dem Zeitpunkt der Kenntnis einer konkreten
            Rechtsverletzung möglich. Bei Bekanntwerden von entsprechenden Rechtsverletzungen werde ich diese Inhalte
            umgehend entfernen.
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Haftung für Links</h2>
          <p>
            Mein Angebot enthält Links zu externen Websites Dritter, auf deren Inhalte ich keinen Einfluss habe. Deshalb
            kann ich für diese fremden Inhalte auch keine Gewähr übernehmen. Für die Inhalte der verlinkten Seiten ist
            stets der jeweilige Anbieter oder Betreiber der Seiten verantwortlich.
          </p>
          <p>
            Die verlinkten Seiten wurden zum Zeitpunkt der Verlinkung auf mögliche Rechtsverstöße überprüft. Rechtswidrige
            Inhalte waren zum Zeitpunkt der Verlinkung nicht erkennbar. Eine permanente inhaltliche Kontrolle der verlinkten
            Seiten ist jedoch ohne konkrete Anhaltspunkte einer Rechtsverletzung nicht zumutbar. Bei Bekanntwerden von
            Rechtsverletzungen werde ich derartige Links umgehend entfernen.
          </p>

          <h2 className="text-2xl font-semibold text-neutral-100 pt-4">Urheberrecht</h2>
          <p>
            Die durch den Seitenbetreiber erstellten Inhalte und Werke auf diesen Seiten unterliegen dem deutschen
            Urheberrecht. Die Vervielfältigung, Bearbeitung, Verbreitung und jede Art der Verwertung außerhalb der Grenzen
            des Urheberrechts bedürfen der schriftlichen Zustimmung des jeweiligen Autors bzw. Erstellers.
          </p>
          <p>
            Soweit die Inhalte auf dieser Seite nicht vom Betreiber erstellt wurden, werden die Urheberrechte Dritter
            beachtet. Insbesondere werden Inhalte Dritter als solche gekennzeichnet. Sollten Sie trotzdem auf eine
            Urheberrechtsverletzung aufmerksam werden, bitte ich um einen entsprechenden Hinweis. Bei Bekanntwerden von
            Rechtsverletzungen werde ich derartige Inhalte umgehend entfernen.
          </p>

          <hr className="border-white/10 my-8" />
          <p className="text-xs text-neutral-500">
            Hinweis: Dieses Impressum wurde als Vorlage/Entwurf erstellt und ersetzt keine individuelle Rechtsberatung.
          </p>
        </section>
      </div>
    </main>
  );
}
