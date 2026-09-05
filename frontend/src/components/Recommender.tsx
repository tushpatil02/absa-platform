/**
 * Slider-driven phone recommender.
 *
 * The sliders are *requirements*, not importance weights: a phone is penalised
 * for falling short and never for exceeding, so an axis left at 1 stops
 * affecting the ranking entirely. That has to be legible from the interface,
 * which is why each slider carries a plain-language reading of its position
 * rather than a bare number, and why a slider at the minimum is visibly marked
 * as "any".
 *
 * Nothing here computes a score. Ranking happens server-side in
 * ml/recommender/similarity.py, so the numbers on screen are the numbers the
 * pipeline produced.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "../api/client";
import { AXES } from "../types";
import type { Axis, Match, Preferences, RecommendResponse } from "../types";

const LABELS: Record<Axis, string> = {
  battery: "Battery",
  camera: "Camera",
  price: "Price",
  display: "Display",
  performance: "Processor",
};

const HINTS: Record<Axis, string> = {
  battery: "How long it lasts and how fast it charges",
  camera: "Photo and video quality",
  price: "Higher means you want it cheaper",
  display: "Screen quality, brightness and size",
  performance: "Speed, memory and responsiveness",
};

const DEFAULTS: Preferences = {
  battery: 5,
  camera: 5,
  price: 5,
  display: 5,
  performance: 5,
};

/** Plain reading of a slider position, so the number is never bare. */
function requirementLabel(value: number): string {
  if (value <= 1) return "any";
  if (value <= 3) return "not fussy";
  if (value <= 5) return "decent";
  if (value <= 7) return "good";
  if (value <= 9) return "very good";
  return "the best";
}

/**
 * Shown wherever a simulated phone's score is. Not decoration: these scores come
 * from generated text, and the only condition under which showing them is
 * defensible is that the interface says so every time.
 */
export function SimulatedBadge() {
  return (
    <span
      className="sim-badge"
      title="Reviews for this phone were generated, not collected. No 2025+ phone review corpus is publicly licensed, so these scores are illustrative."
    >
      simulated
    </span>
  );
}

function matchTone(percent: number): string {
  if (percent >= 95) return "match--strong";
  if (percent >= 80) return "match--good";
  if (percent >= 60) return "match--fair";
  return "match--weak";
}

function MatchCard({
  match,
  rank,
  onOpen,
}: {
  match: Match;
  rank: number;
  onOpen: (modelKey: string) => void;
}) {
  const { phone } = match;
  const worst = match.worst_axis as Axis | null;

  return (
    <li className="match-card">
      <div className="match-card__rank">{rank}</div>

      {phone.image ? (
        // Amazon's CDN can 404 on old listings; hide rather than show a broken icon.
        <img
          className="match-card__image"
          src={phone.image}
          alt=""
          loading="lazy"
          onError={(event) => {
            event.currentTarget.style.display = "none";
          }}
        />
      ) : null}

      <div className="match-card__body">
        <span className="match-card__title">
          <button className="match-card__name" onClick={() => onOpen(phone.model_key)}>
            {phone.name}
          </button>
          {phone.simulated && <SimulatedBadge />}
        </span>
        <div className="match-card__meta">
          {phone.price != null && <span>${phone.price.toFixed(0)}</span>}
          <span>{phone.reviews_total.toLocaleString()} reviews</span>
          {phone.avg_rating != null && <span>{phone.avg_rating.toFixed(1)}★ on Amazon</span>}
        </div>

        <div className="aspect-bars">
          {AXES.map((axis) => {
            const score = phone.aspects.find((a) => a.aspect === axis);
            if (!score) return null;
            const missed = (match.shortfalls[axis] ?? 0) > 0;
            return (
              <div className="aspect-bar" key={axis}>
                <span className="aspect-bar__label">{LABELS[axis]}</span>
                <span className="aspect-bar__track">
                  <span
                    className={`aspect-bar__fill${missed ? " aspect-bar__fill--short" : ""}`}
                    style={{ width: `${((score.score - 1) / 9) * 100}%` }}
                  />
                </span>
                <span className="aspect-bar__value">{score.score.toFixed(1)}</span>
              </div>
            );
          })}
        </div>
      </div>

      <div className={`match-card__score ${matchTone(match.match_percent)}`}>
        <strong>{match.match_percent.toFixed(0)}%</strong>
        <span>match</span>
        {worst ? (
          <em className="match-card__gap">
            short on {LABELS[worst]} by {match.shortfalls[worst].toFixed(1)}
          </em>
        ) : (
          <em className="match-card__gap">meets everything</em>
        )}
      </div>
    </li>
  );
}

export function Recommender({ onOpen }: { onOpen: (modelKey: string) => void }) {
  const [preferences, setPreferences] = useState<Preferences>(DEFAULTS);
  const [includeSimulated, setIncludeSimulated] = useState(true);
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Dragging a slider fires continuously; without debouncing every pixel of
  // travel would become a request.
  const timer = useRef<number | undefined>(undefined);

  const fetchMatches = useCallback((next: Preferences, withSimulated: boolean) => {
    setLoading(true);
    api
      .recommend(next, 12, withSimulated)
      .then((response) => {
        setResult(response);
        setError(null);
      })
      .catch((cause: unknown) => {
        setError(cause instanceof ApiError ? cause.message : "Something went wrong.");
        setResult(null);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(
      () => fetchMatches(preferences, includeSimulated),
      250,
    );
    return () => window.clearTimeout(timer.current);
  }, [preferences, includeSimulated, fetchMatches]);

  const update = (axis: Axis, value: number) =>
    setPreferences((current) => ({ ...current, [axis]: value }));

  return (
    <div className="recommender">
      <section className="card sliders" aria-labelledby="sliders-heading">
        <div className="card__head">
          <h2 id="sliders-heading">What matters to you?</h2>
          <button
            className="button button--ghost button--sm"
            onClick={() => setPreferences(DEFAULTS)}
          >
            Reset
          </button>
        </div>
        <p className="sliders__intro">
          Each slider is a <strong>minimum</strong>, not a weight. A phone is never
          penalised for being better than you asked, so leaving one at{" "}
          <strong>1</strong> means you don&rsquo;t care about it.
        </p>

        {AXES.map((axis) => (
          <div className="slider" key={axis}>

            <label className="slider__label" htmlFor={`slider-${axis}`}>
              <span className="slider__name">
                {LABELS[axis]}
                {axis === "price" && (
                  <span className="slider__badge" title="From the listed price, not review sentiment">
                    listed price
                  </span>
                )}
              </span>
              <span className="slider__value">
                {preferences[axis].toFixed(0)}
                <em>{requirementLabel(preferences[axis])}</em>
              </span>
            </label>
            <input
              id={`slider-${axis}`}
              className="slider__input"
              type="range"
              min={1}
              max={10}
              step={1}
              value={preferences[axis]}
              onChange={(event) => update(axis, Number(event.target.value))}
              aria-describedby={`hint-${axis}`}
            />
            <span className="slider__hint" id={`hint-${axis}`}>
              {HINTS[axis]}
              {axis === "price" && result?.price_target != null
                ? ` — around $${result.price_target.toFixed(0)}`
                : ""}
            </span>
          </div>
        ))}
      </section>

      <section className="results" aria-live="polite">
        {result && result.simulated_considered > 0 && includeSimulated && (
          <div className="sim-notice">
            <strong>{result.simulated_considered}</strong> of the{" "}
            {result.considered} ranked phones are <SimulatedBadge /> — 2025–2026
            models whose reviews were generated, because no 2025+ review corpus
            is publicly licensed. Their scores are illustrative and are excluded
            from the reliability checks.{" "}
            <button className="link-button" onClick={() => setIncludeSimulated(false)}>
              Show only real phones
            </button>
          </div>
        )}
        {!includeSimulated && (
          <div className="sim-notice">
            Showing only phones with real, collected reviews.{" "}
            <button className="link-button" onClick={() => setIncludeSimulated(true)}>
              Include simulated 2025–2026 phones
            </button>
          </div>
        )}
        {error && (
          <div className="alert alert--error" role="alert">
            <span className="alert__icon">!</span>
            <span>{error}</span>
          </div>
        )}

        {result && !error && (
          <>
            <div className="results__head">
              <h2>
                {result.matches.length} best {result.matches.length === 1 ? "match" : "matches"}
              </h2>
              <span className="results__meta">
                ranked from {result.considered} phones
                {loading && <span className="spinner" aria-label="Updating" />}
              </span>
            </div>

            {result.matches.every((m) => m.match_percent === 100) && (
              <p className="results__note">
                Every phone meets these requirements, so they are ordered by overall
                score. Raise a slider to separate them.
              </p>
            )}

            <ol className="match-list">
              {result.matches.map((match, index) => (
                <MatchCard
                  key={match.phone.model_key}
                  match={match}
                  rank={index + 1}
                  onOpen={onOpen}
                />
              ))}
            </ol>
          </>
        )}

        {!result && !error && (
          <p className="empty">
            <span className="spinner" /> Finding matches…
          </p>
        )}
      </section>
    </div>
  );
}
