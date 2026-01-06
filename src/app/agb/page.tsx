import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AGB",
};

export default function AgbPage() {
  return (
    <main className="min-h-screen bg-black/[0.96] text-neutral-100 relative overflow-hidden">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <h1 className="text-4xl font-bold mb-6">Allgemeine Geschäftsbedingungen (AGB)</h1>

        <div className="space-y-6 text-neutral-200">
          <p className="text-sm text-neutral-400">Stand: 06.01.2026</p>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">1. Geltungsbereich</h2>
            <p>
              Diese Allgemeinen Geschäftsbedingungen (nachfolgend „AGB“) gelten für sämtliche Verträge über die Nutzung
              der Online-Plattform / Webanwendung „Open Hallucination Index“ (nachfolgend „Dienst“), einschließlich des
              Erwerbs von Token-Paketen („OHI-Tokens“) als digitale Nutzungs-/Verbrauchseinheiten innerhalb des Dienstes.
            </p>
            <p>
              Abweichende, entgegenstehende oder ergänzende Bedingungen von Kund:innen werden nur Vertragsbestandteil, wenn
              ich ihrer Geltung ausdrücklich schriftlich zugestimmt habe.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">2. Anbieter / Vertragspartner</h2>
            <p>
              Vertragspartner und Anbieter des Dienstes ist:
              <br />
              shiftbloom studio.<br />
              Fabian Zimber<br />
              Up de Worth 6a<br />
              22927 Großhansdorf<br />
              Deutschland<br />
              E-Mail: hi@shiftbloom.studio
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">3. Leistungsbeschreibung</h2>
            <p>
              Der Dienst stellt Funktionen bereit, mit denen registrierte Nutzer:innen Inhalte prüfen/analysieren und
              Ergebnisse in ihrem Account verwalten können. Bestimmte Funktionen setzen den Verbrauch von OHI-Tokens
              voraus.
            </p>
            <p>
              OHI-Tokens sind keine gesetzliche Währung und keine Kryptowährung. Sie dienen ausschließlich der Nutzung
              von Funktionen innerhalb des Dienstes.
            </p>
            <p>
              Ich schulde die Bereitstellung des Dienstes im Rahmen des aktuellen technischen Stands. Ein bestimmter Erfolg
              (z.B. bestimmte Analyseergebnisse, „Fehlerfreiheit“ oder Eignung für einen bestimmten Zweck) wird nicht
              geschuldet.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">4. Registrierung und Nutzerkonto</h2>
            <p>
              Für die Nutzung bestimmter Funktionen ist die Erstellung eines Nutzerkontos erforderlich. Bei der
              Registrierung sind wahrheitsgemäße Angaben zu machen. Zugangsdaten sind vertraulich zu behandeln; eine
              Weitergabe an Dritte ist untersagt.
            </p>
            <p>
              Ich behalte mir vor, Nutzerkonten zu sperren oder zu löschen, wenn konkrete Anhaltspunkte für Missbrauch,
              Sicherheitsrisiken oder erhebliche Verstöße gegen diese AGB bestehen.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">5. Vertragsschluss (Token-Kauf)</h2>
            <p>
              Die Darstellung von Token-Paketen im Dienst stellt kein rechtlich bindendes Angebot dar, sondern eine
              Aufforderung zur Abgabe einer Bestellung.
            </p>
            <p>
              Der Vertrag über den Erwerb eines Token-Pakets kommt zustande, wenn der Zahlungsprozess abgeschlossen wird
              und die Zahlungsbestätigung/der Checkout erfolgreich ist.
            </p>
            <p>
              Die Abwicklung der Zahlung erfolgt über einen externen Zahlungsdienstleister (derzeit: Stripe). Es gelten
              ergänzend die Bedingungen des Zahlungsdienstleisters.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">6. Preise, Steuern, Zahlung</h2>
            <p>
              Es gelten die zum Zeitpunkt der Bestellung im Checkout angezeigten Preise. Sofern Umsatzsteuer anfällt und
              auszuweisen ist, wird diese im Checkout entsprechend ausgewiesen. (Hinweis: Bitte steuerliche Angaben/Status
              prüfen und ggf. anpassen.)
            </p>
            <p>
              Die Zahlung ist sofort fällig. Akzeptierte Zahlungsmethoden ergeben sich aus dem Checkout.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">7. Bereitstellung, Gutschrift der Tokens</h2>
            <p>
              Nach erfolgreichem Zahlungseingang werden die erworbenen OHI-Tokens dem Nutzerkonto gutgeschrieben.
              Die Gutschrift erfolgt in der Regel automatisiert; im Ausnahmefall kann es zu Verzögerungen kommen.
            </p>
            <p>
              Bei technischen Problemen bitte ich um Kontaktaufnahme unter hi@shiftbloom.studio unter Angabe der
              Bestell-/Zahlungsinformationen.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">8. Widerrufsrecht (Verbraucher:innen)</h2>
            <p>
              Verbraucher:innen steht grundsätzlich ein gesetzliches Widerrufsrecht zu. Dieses kann bei digitalen Inhalten
              unter bestimmten Voraussetzungen erlöschen.
            </p>
            <p>
              Erlöschen des Widerrufsrechts: Das Widerrufsrecht kann erlöschen, wenn (1) ich mit der Ausführung des
              Vertrags begonnen habe, nachdem Sie (2) ausdrücklich zugestimmt haben, dass ich vor Ablauf der Widerrufsfrist
              mit der Ausführung beginne, und (3) Sie Ihre Kenntnis davon bestätigt haben, dass Sie durch Ihre Zustimmung
              mit Beginn der Ausführung Ihr Widerrufsrecht verlieren.
            </p>
            <p>
              Hinweis: Ob und wie diese Zustimmung technisch im Checkout eingeholt wird, sollte überprüft und ggf.
              ergänzt werden, um rechtlich sauber zu sein.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">9. Nutzungsrechte und erlaubte Nutzung</h2>
            <p>
              Sofern der Dienst Ausgaben/Ergebnisse erzeugt, erhalten Nutzer:innen hieran ein einfaches, nicht exklusives,
              nicht übertragbares Nutzungsrecht für eigene Zwecke. Eine Weitergabe, Veröffentlichung oder kommerzielle
              Verwertung kann je nach Inhalt/Quelle eingeschränkt sein.
            </p>
            <p>
              Untersagt sind insbesondere: (a) rechtswidrige Inhalte, (b) Umgehung von Sicherheitsmaßnahmen,
              (c) automatisierte Massennutzung, die den Dienst beeinträchtigt, (d) Reverse Engineering, soweit gesetzlich
              nicht ausdrücklich erlaubt.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">10. Verfügbarkeit, Wartung</h2>
            <p>
              Ich bemühe mich um eine hohe Verfügbarkeit, schulde jedoch keine ununterbrochene Verfügbarkeit. Wartungen,
              Sicherheitsupdates und technische Störungen können zu temporären Einschränkungen führen.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">11. Gewährleistung</h2>
            <p>
              Es gelten die gesetzlichen Gewährleistungsrechte. Bei digitalen Leistungen können Fehler auftreten; ich
              werde berechtigte Mängelrügen im Rahmen der gesetzlichen Vorgaben bearbeiten.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">12. Haftung</h2>
            <p>
              Ich hafte unbeschränkt bei Vorsatz und grober Fahrlässigkeit sowie bei Verletzung des Lebens, des Körpers
              oder der Gesundheit.
            </p>
            <p>
              Bei einfacher Fahrlässigkeit hafte ich nur bei Verletzung wesentlicher Vertragspflichten (Kardinalpflichten)
              und beschränkt auf den vertragstypischen, vorhersehbaren Schaden.
            </p>
            <p>
              Die Haftung nach dem Produkthaftungsgesetz bleibt unberührt.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">13. Laufzeit, Kündigung</h2>
            <p>
              Nutzerkonten können von Nutzer:innen jederzeit beendet werden (z.B. per E-Mail). Bereits erworbene Tokens
              können abhängig von der technischen Ausgestaltung ggf. verfallen; hierzu bitte konkrete Regeln festlegen.
              (Platzhalter: bitte an eure Token-Logik anpassen.)
            </p>
            <p>
              Das Recht zur außerordentlichen Kündigung aus wichtigem Grund bleibt unberührt.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">14. Datenschutz</h2>
            <p>
              Informationen zur Verarbeitung personenbezogener Daten finden sich in der Datenschutzerklärung.
            </p>
          </section>

          <section className="space-y-3">
            <h2 className="text-2xl font-semibold text-neutral-100">15. Schlussbestimmungen</h2>
            <p>
              Es gilt das Recht der Bundesrepublik Deutschland unter Ausschluss des UN-Kaufrechts. Sind Sie Verbraucher:in,
              gilt diese Rechtswahl nur, soweit dadurch nicht der Schutz zwingender Bestimmungen des Rechts des Staates,
              in dem Sie Ihren gewöhnlichen Aufenthalt haben, entzogen wird.
            </p>
            <p>
              Sofern Sie Kaufmann/Kauffrau, juristische Person des öffentlichen Rechts oder öffentlich-rechtliches
              Sondervermögen sind, ist Gerichtsstand – soweit zulässig – der Sitz des Anbieters.
            </p>
            <p>
              Sollten einzelne Bestimmungen dieser AGB ganz oder teilweise unwirksam sein, bleibt die Wirksamkeit der
              übrigen Bestimmungen unberührt.
            </p>
          </section>

          <hr className="border-white/10 my-8" />
          <p className="text-xs text-neutral-500">
            Hinweis: Diese AGB wurden als Vorlage/Entwurf erstellt und ersetzen keine individuelle Rechtsberatung.
            Bitte insbesondere Widerruf/Steuern/Token-Verfall an eure tatsächlichen Abläufe anpassen.
          </p>
        </div>
      </div>
    </main>
  );
}
