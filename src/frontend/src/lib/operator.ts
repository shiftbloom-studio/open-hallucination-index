/**
 * Operator details shown on the legal pages (Impressum, privacy, terms).
 *
 * The frontend is a static export, so these are read from NEXT_PUBLIC_*
 * variables at build time. Set them in your deploy environment; the
 * repository only ships placeholders. See src/frontend/.env.example.
 */
export const operator = {
  studio: process.env.NEXT_PUBLIC_OPERATOR_STUDIO ?? "shiftbloom studio.",
  name: process.env.NEXT_PUBLIC_OPERATOR_NAME ?? "",
  street: process.env.NEXT_PUBLIC_OPERATOR_STREET ?? "",
  city: process.env.NEXT_PUBLIC_OPERATOR_CITY ?? "",
  country: process.env.NEXT_PUBLIC_OPERATOR_COUNTRY ?? "Germany",
  email: process.env.NEXT_PUBLIC_OPERATOR_EMAIL ?? "hi@shiftbloom.studio",
} as const;

/** "shiftbloom studio., operated by Jane Doe" or just the studio name. */
export const operatorLabel = operator.name
  ? `${operator.studio}, operated by ${operator.name}`
  : operator.studio;
