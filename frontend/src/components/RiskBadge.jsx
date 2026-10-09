/** A risk level as icon + word + description, never colour alone. `level` comes from risk.js. */
export default function RiskBadge({ level, size = "lg" }) {
  const sizes = size === "lg" ? "gap-2 px-3 py-1.5 text-lg sm:text-xl" : "gap-1.5 px-2 py-0.5 text-sm";
  return (
    <span className={`inline-flex shrink-0 items-center rounded-lg font-extrabold tracking-wide ring-2 ${sizes} ${level.badge}`}>
      <span aria-hidden="true">{level.icon}</span>
      <span>{level.label}</span>
      {size === "lg" && <span className="text-sm font-semibold">{level.description}</span>}
    </span>
  );
}
