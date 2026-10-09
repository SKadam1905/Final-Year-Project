import { useEffect, useRef, useState } from "react";

const reducedMotion = () => typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;

/** A number that eases to each new value (ease-out cubic), so live counters glide rather than jump.
 * Jumps straight there with "reduce motion" on, or where animation frames are not available. */
export function useAnimatedNumber(value, duration = 700) {
  const [shown, setShown] = useState(value);
  const current = useRef(value);

  useEffect(() => {
    if (value == null || current.current == null || reducedMotion() || typeof requestAnimationFrame !== "function") {
      current.current = value;
      setShown(value);
      return undefined;
    }
    const from = current.current;
    const start = performance.now();
    let frame;
    const step = (t) => {
      const k = Math.min(1, (t - start) / duration);
      const next = from + (value - from) * (1 - (1 - k) ** 3);
      current.current = next;
      setShown(next);
      if (k < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [value, duration]);

  return shown;
}
