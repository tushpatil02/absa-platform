import type { AspectSentiment, Polarity } from "../types";

/**
 * The 1–10 meter.
 *
 * The track is the polarity axis itself (negative pole → neutral → positive
 * pole) and the marker sits at the model's score. Showing the whole axis rather
 * than a filled bar matters: a filled bar reads as "how much", but this scale
 * has no zero — 5.5 is the neutral midpoint, not "half as good".
 */
export function ScoreMeter({ score, min = 1, max = 10 }: { score: number; min?: number; max?: number }) {
  const clamped = Math.min(Math.max(score, min), max);
  const percent = ((clamped - min) / (max - min)) * 100;

  return (
    <div className="meter">
      <div
        className="meter__track"
        role="meter"
        aria-valuenow={Number(score.toFixed(2))}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-label={`Sentiment score ${score.toFixed(2)} out of ${max}`}
      >
        <span className="meter__marker" style={{ left: `${percent}%` }} />
      </div>
      <div className="meter__scale" aria-hidden="true">
        <span>{min} Very negative</span>
        <span>{(min + max) / 2} Neutral</span>
        <span>{max} Very positive</span>
      </div>
    </div>
  );
}

export function PolarityPill({ polarity }: { polarity: Polarity }) {
  return (
    <span className={`pill pill--${polarity}`}>
      {polarity.charAt(0).toUpperCase() + polarity.slice(1)}
    </span>
  );
}

/**
 * One detected aspect.
 *
 * Score and confidence are shown as separate figures on purpose: "how positive"
 * and "how sure" are different questions, and blending them would make both
 * unreadable. A 5.5 from a confident neutral and a 5.5 from a coin flip look
 * identical in the score alone — the confidence field is what distinguishes them.
 */
export function AspectCard({ aspect }: { aspect: AspectSentiment }) {
  return (
    <li className="aspect">
      <div className="aspect__head">
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span className="aspect__name">{aspect.display_name}</span>
          <PolarityPill polarity={aspect.polarity} />
        </div>
        <div>
          <span className="aspect__score">{aspect.score.toFixed(1)}</span>
          <span className="aspect__score-max"> / 10</span>
        </div>
      </div>

      <ScoreMeter score={aspect.score} />

      <div className="aspect__meta">
        <span>{aspect.label}</span>
        <span title="Probability of the predicted polarity">
          Sentiment confidence {(aspect.confidence * 100).toFixed(0)}%
        </span>
        <span title="Probability that this aspect is discussed in the review">
          Aspect detected {(aspect.detection_confidence * 100).toFixed(0)}%
        </span>
      </div>
    </li>
  );
}
