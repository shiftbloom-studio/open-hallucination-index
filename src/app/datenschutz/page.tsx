import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Datenschutzerklärung",
};

export default function DatenschutzPage() {
  return (
    <main className="min-h-screen bg-black/[0.96] text-neutral-100 relative overflow-hidden">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <h1 className="text-4xl font-bold mb-6">Datenschutzerklärung</h1>

        <div className="space-y-6 text-neutral-200">
          <p className="text-sm text-neutral-400">Stand: 06.01.2026</p>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">1. Verantwortlicher</h2>
            <p>
              Verantwortlich im Sinne der Datenschutz-Grundverordnung (DSGVO) ist:
              <br />
              shiftbloom studio.<br />
              Fabian Zimber<br />
              Up de Worth 6a, 22927 Großhansdorf, Deutschland<br />
              E-Mail: hi@shiftbloom.studio
            </p>
            <p>
              (Hinweis: Falls ein Datenschutzbeauftragter benannt ist, bitte die Kontaktdaten ergänzen.)
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">2. Überblick: Welche Daten verarbeiten wir?</h2>
            <p>
              Je nach Nutzung des Dienstes verarbeiten wir insbesondere:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Bestandsdaten (z.B. E-Mail, Konto-ID)</li>
              <li>Nutzungs-/Metadaten (z.B. Login-Status, Token-Stand, technisch erforderliche Kennungen)</li>
              <li>Inhaltsdaten, die Sie im Dienst eingeben/hochladen (z.B. Texte zur Analyse)</li>
              <li>Zahlungs-/Transaktionsdaten (im Zusammenhang mit Stripe-Checkout)</li>
              <li>Protokolldaten (z.B. IP-Adresse, Zeitpunkt, Request-Informationen in Server-Logs)</li>
              <li>Kommunikationsdaten (z.B. E-Mail-Inhalte bei Support-Anfragen)</li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">3. Zwecke und Rechtsgrundlagen</h2>
            <p>
              Wir verarbeiten personenbezogene Daten nur, soweit dies erlaubt ist. Typische Zwecke und Rechtsgrundlagen
              sind:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>
                Vertragserfüllung und Durchführung vorvertraglicher Maßnahmen (Art. 6 Abs. 1 lit. b DSGVO), z.B.
                Registrierung, Login, Bereitstellung des Dienstes, Token-Gutschrift.
              </li>
              <li>
                Rechtliche Verpflichtungen (Art. 6 Abs. 1 lit. c DSGVO), z.B. handels-/steuerrechtliche Aufbewahrung.
              </li>
              <li>
                Berechtigte Interessen (Art. 6 Abs. 1 lit. f DSGVO), z.B. IT-Sicherheit, Missbrauchsprävention,
                Fehleranalyse, Betrieb und Optimierung.
              </li>
              <li>
                Einwilligung (Art. 6 Abs. 1 lit. a DSGVO), sofern wir optionale Cookies/Tracking einsetzen.
                (Derzeit sind im Code keine typischen Tracking-Tools ersichtlich; bitte prüfen, ob tatsächlich welche
                eingesetzt werden.)
              </li>
            </ul>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">4. Hosting und Server-Logs</h2>
            <p>
              Beim Besuch der Website werden durch den Hosting-Provider bzw. durch die Server-Infrastruktur technisch
              notwendige Daten verarbeitet und in Logfiles gespeichert (z.B. IP-Adresse, Datum/Uhrzeit, aufgerufene Seite,
              Referrer-URL, User-Agent, Statuscodes). Die Verarbeitung erfolgt zur Auslieferung der Website, zur
              Gewährleistung der IT-Sicherheit und zur Fehleranalyse.
            </p>
            <p>
              Rechtsgrundlage ist Art. 6 Abs. 1 lit. f DSGVO (berechtigtes Interesse an sicherem und stabilem Betrieb).
              (Hinweis: Bitte den konkreten Hosting-Anbieter und ggf. Auftragsverarbeitungsvertrag/Datenstandort ergänzen.)
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">5. Authentifizierung und Nutzerkonto (Supabase)</h2>
            <p>
              Für Registrierung/Login nutzen wir Supabase. Dabei werden u.a. Ihre E-Mail-Adresse, technische Kennungen und
              Session-Informationen verarbeitet. Zur Aufrechterhaltung Ihrer Anmeldung setzt Supabase technisch erforderliche
              Cookies/Token (Session-Cookies).
            </p>
            <p>
              Rechtsgrundlage ist Art. 6 Abs. 1 lit. b DSGVO (Vertragserfüllung) sowie Art. 6 Abs. 1 lit. f DSGVO
              (Sicherheit und Missbrauchsprävention).
            </p>
            <p>
              Empfänger: Supabase (je nach Konfiguration ggf. auch in Drittländern). Sofern eine Übermittlung in ein
              Drittland erfolgt, stützt sich diese – soweit erforderlich – auf geeignete Garantien (z.B.
              Standardvertragsklauseln). Details entnehmen Sie bitte den Datenschutzhinweisen von Supabase.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">6. Zahlungsabwicklung (Stripe)</h2>
            <p>
              Für den Erwerb von Token-Paketen nutzen wir Stripe als Zahlungsdienstleister. Wenn Sie einen Kauf tätigen,
              werden Sie zum Stripe-Checkout weitergeleitet. Dabei verarbeitet Stripe Zahlungsdaten (z.B. Kartendaten), die
              wir nicht vollständig einsehen.
            </p>
            <p>
              Wir übermitteln an Stripe insbesondere:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>E-Mail-Adresse (zur Zuordnung/Belegkommunikation)</li>
              <li>technische Kennungen (z.B. Session/Transaktionsdaten)</li>
              <li>Metadaten zur Zuordnung im Dienst (z.B. Nutzer-ID, Paket-ID)</li>
            </ul>
            <p>
              Rechtsgrundlage ist Art. 6 Abs. 1 lit. b DSGVO (Zahlungsabwicklung/Vertragserfüllung) sowie ggf.
              Art. 6 Abs. 1 lit. c DSGVO (Aufbewahrungspflichten).
            </p>
            <p>
              Stripe kann Daten in Drittländer (z.B. USA) übermitteln. Soweit erforderlich, erfolgt die Übermittlung unter
              geeigneten Garantien (z.B. Standardvertragsklauseln). Details finden Sie in den Datenschutzhinweisen von Stripe.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">7. Datenbank / Speicherung</h2>
            <p>
              Zur Bereitstellung des Dienstes speichern wir Daten in einer Datenbank (z.B. Nutzerkonto, Token-Stand,
              Einstellungen). Die Speicherung erfolgt so lange, wie dies für die Vertragserfüllung erforderlich ist oder
              gesetzliche Aufbewahrungspflichten bestehen.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">8. Cookies</h2>
            <p>
              Wir setzen Cookies ein, soweit diese technisch erforderlich sind (z.B. für Login/Sitzung). Technisch
              erforderliche Cookies sind notwendig, damit der Dienst funktioniert.
            </p>
            <p>
              Soweit wir darüber hinaus optionale Cookies (z.B. für Statistik/Marketing) einsetzen, erfolgt dies nur mit
              Ihrer Einwilligung (Art. 6 Abs. 1 lit. a DSGVO). (Hinweis: Im aktuellen Code sind keine typischen
              Analytics/Tracking-Integrationen ersichtlich; falls doch, bitte konkret ergänzen.)
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">9. Kontaktaufnahme</h2>
            <p>
              Wenn Sie uns per E-Mail kontaktieren, verarbeiten wir Ihre Angaben zur Bearbeitung der Anfrage.
              Rechtsgrundlage ist Art. 6 Abs. 1 lit. b DSGVO (Anbahnung/Erfüllung) oder Art. 6 Abs. 1 lit. f DSGVO
              (berechtigtes Interesse an effizienter Kommunikation).
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">10. Empfänger, Auftragsverarbeiter</h2>
            <p>
              Wir setzen Dienstleister ein (z.B. Hosting, Supabase, Stripe), die personenbezogene Daten in unserem Auftrag
              verarbeiten. Mit diesen Dienstleistern schließen wir – soweit erforderlich – Verträge zur
              Auftragsverarbeitung (Art. 28 DSGVO).
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">11. Speicherdauer</h2>
            <p>
              Wir speichern personenbezogene Daten nur so lange, wie es für die jeweiligen Zwecke erforderlich ist.
              Darüber hinaus speichern wir Daten, soweit gesetzliche Aufbewahrungspflichten bestehen (z.B. steuer-/handelsrechtlich).
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">12. Ihre Rechte</h2>
            <p>
              Ihnen stehen – je nach gesetzlichen Voraussetzungen – folgende Rechte zu:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Auskunft (Art. 15 DSGVO)</li>
              <li>Berichtigung (Art. 16 DSGVO)</li>
              <li>Löschung (Art. 17 DSGVO)</li>
              <li>Einschränkung der Verarbeitung (Art. 18 DSGVO)</li>
              <li>Datenübertragbarkeit (Art. 20 DSGVO)</li>
              <li>Widerspruch gegen Verarbeitungen (Art. 21 DSGVO)</li>
              <li>Widerruf erteilter Einwilligungen (Art. 7 Abs. 3 DSGVO) mit Wirkung für die Zukunft</li>
              <li>Beschwerde bei einer Aufsichtsbehörde (Art. 77 DSGVO)</li>
            </ul>
            <p>
              Zur Geltendmachung Ihrer Rechte genügt eine Nachricht an hi@shiftbloom.studio.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">13. Pflicht zur Bereitstellung von Daten</h2>
            <p>
              Für die Registrierung und Nutzung des Dienstes ist die Bereitstellung bestimmter Daten (z.B. E-Mail) erforderlich.
              Ohne diese Daten kann der Dienst nicht oder nur eingeschränkt genutzt werden.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">14. Änderungen dieser Datenschutzerklärung</h2>
            <p>
              Wir können diese Datenschutzerklärung anpassen, wenn sich Rechtslage, Dienste oder Datenverarbeitungen ändern.
              Es gilt die jeweils aktuelle Fassung.
            </p>
          </section>

          <hr className="border-white/10 my-8" />
          <p className="text-xs text-neutral-500">
            Hinweis: Diese Datenschutzerklärung wurde als Vorlage/Entwurf erstellt und ersetzt keine individuelle Rechtsberatung.
            Bitte insbesondere Hosting-Anbieter, konkrete Cookies/Tracking und Datenstandorte final prüfen und eintragen.
          </p>
        </div>
      </div>
    </main>
  );
}
