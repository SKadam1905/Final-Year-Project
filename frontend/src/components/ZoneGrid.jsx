import ZoneCard from "./ZoneCard.jsx";

/** Zones, already sorted highest risk first. */
export default function ZoneGrid({ zones, now, connected, onSelect, countsAvailable = true }) {
  return (
    <ul className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-3">
      {zones.map((zone) => (
        <li key={zone.id}>
          <ZoneCard zone={zone} now={now} connected={connected} onSelect={onSelect} countsAvailable={countsAvailable} />
        </li>
      ))}
    </ul>
  );
}
