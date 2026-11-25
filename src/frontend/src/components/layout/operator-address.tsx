import { operator } from "@/lib/operator";

/**
 * Operator postal address for legal pages. Values come from build-time
 * environment variables (see src/frontend/.env.example), never from source.
 */
export function OperatorAddress({ inline = false }: { inline?: boolean }) {
  const lines = [operator.studio, operator.name, operator.street, operator.city, operator.country].filter(Boolean);
  if (inline) {
    return <>{lines.join(", ")}</>;
  }
  return (
    <>
      {lines.map((line) => (
        <span key={line}>
          {line}
          <br />
        </span>
      ))}
    </>
  );
}
