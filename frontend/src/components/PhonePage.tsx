/**
 * One phone: its aspect profile, the sentences behind each score, and a form
 * to submit a review and see it analysed.
 *
 * Two honesty requirements shape this page.
 *
 * **Every score is traceable.** Each aspect shows the sentences that produced
 * it, positive and negative, taken from the model's own per-sentence output.
 * A number a reader cannot check is a number they have to trust.
 *
 * **A submitted review does not move the published scores.** It is analysed and
 * shown immediately, but the phone's profile is built from hundreds of reviews
 * and does not shift because one person typed a sentence. The page says so,
 * rather than letting the reader assume otherwise.
 */

import { useEffect, useState } from "react";
import { ApiError, api } from "../api/client";
import { AXES } from "../types";
import type { Axis, Evidence, PhoneDetail, SubmitReviewResponse } from "../types";

const LABELS: Record<Axis, string> = {
  battery: "Battery",
  camera: "Camera",
  price: "Price",
  display: "Display",
  performance: "Processor",
};

function scoreTone(score: number): string {
  if (score >= 7.5) return "score--positive";
  if (score >= 4.5) return "score--neutral";
  return "score--negative";
}

function EvidenceList({ items }: { items: Evidence[] }) {
  if (!items.length) {
    return <p className="muted">No example sentences were recorded for this aspect.</p>;
  }
  return (
    <ul className="evidence">
      {items.map((item, index) => (
        <li className={`evidence__item evidence__item--${item.polarity}`} key={index}>
          {/* The aspect label is not decoration. One sentence can support two
              aspects ("the camera is awesome and battery life is great"), so
              without it the same quote appears twice looking like a bug. */}
          <span className="evidence__aspect">{item.display_name}</span>
          <span className="evidence__score">{item.score.toFixed(1)}</span>
          <q className="evidence__text">{item.sentence}</q>
        </li>
      ))}
    </ul>
  );
}

function ReviewForm({ modelKey }: { modelKey: string }) {
  const [text, setText] = useState("");
  const [result, setResult] = useState<SubmitReviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    setError(null);
    api
      .submitReview(modelKey, text)
      .then((response) => {
        setResult(response);
        setText("");
      })
      .catch((cause: unknown) =>
        setError(cause instanceof ApiError ? cause.message : "Could not submit the review."),
      )
      .finally(() => setBusy(false));
  };

  return (
    <section className="card" aria-labelledby="submit-heading">
      <div className="card__head">
        <h2 id="submit-heading">Write a review</h2>
      </div>
      <p className="muted">
        Your review is analysed sentence by sentence and shown below. It is stored,
        but it does <strong>not</strong> change the scores above &mdash; those come
        from hundreds of reviews and should not move because of one.
      </p>

      <form onSubmit={submit}>
        <textarea
          className="textarea"
          rows={4}
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="The battery easily lasts a day, but photos are grainy indoors."
          maxLength={5000}
          aria-label="Your review"
        />
        <div className="form-row">
          <button className="button" type="submit" disabled={busy || !text.trim()}>
            {busy ? "Analysing…" : "Analyse my review"}
          </button>
          <span className="muted">{text.length}/5000</span>
        </div>
      </form>

      {error && (
        <div className="alert alert--error" role="alert">
          <span className="alert__icon">!</span>
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="submitted">
          <h3>What the model found</h3>
          {result.aspects.length === 0 ? (
            <p className="muted">No aspect was detected confidently in that text.</p>
          ) : (
            <>
              <ul className="chips">
                {result.aspects.map((aspect) => (
                  <li key={aspect.aspect} className={`chip ${scoreTone(aspect.score)}`}>
                    {aspect.display_name} {aspect.score.toFixed(1)}
                  </li>
                ))}
              </ul>
              <EvidenceList items={result.evidence} />
            </>
          )}
          <p className="muted small">Model: {result.model}</p>
        </div>
      )}
    </section>
  );
}

export function PhonePage({ modelKey, onBack }: { modelKey: string; onBack: () => void }) {
  const [phone, setPhone] = useState<PhoneDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // No state reset here: App gives this component a `key` of the model key,
    // so switching phones remounts it with fresh state. Resetting inside the
    // effect instead would render the previous phone's data for one frame.
    api
      .phone(modelKey)
      .then(setPhone)
      .catch((cause: unknown) =>
        setError(cause instanceof ApiError ? cause.message : "Could not load this phone."),
      );
  }, [modelKey]);

  if (error) {
    return (
      <div className="alert alert--error" role="alert">
        <span className="alert__icon">!</span>
        <span>{error}</span>
      </div>
    );
  }

  if (!phone) {
    return (
      <p className="empty">
        <span className="spinner" /> Loading…
      </p>
    );
  }

  return (
    <div className="phone-page">
      <button className="button button--ghost button--sm" onClick={onBack}>
        &larr; Back to results
      </button>

      <section className="card phone-hero">
        {phone.image ? (
          <img
            className="phone-hero__image"
            src={phone.image}
            alt=""
            onError={(event) => {
              event.currentTarget.style.display = "none";
            }}
          />
        ) : null}
        <div>
          <h1>{phone.name}</h1>
          <div className="phone-hero__meta">
            {phone.brand && <span>{phone.brand}</span>}
            {phone.price != null && <span>${phone.price.toFixed(0)}</span>}
            {phone.avg_rating != null && <span>{phone.avg_rating.toFixed(1)}★ on Amazon</span>}
            <span>
              {phone.reviews_scored.toLocaleString()} of{" "}
              {phone.reviews_total.toLocaleString()} reviews analysed
            </span>
          </div>
        </div>
      </section>

      <section className="card" aria-labelledby="profile-heading">
        <div className="card__head">
          <h2 id="profile-heading">Aspect profile</h2>
        </div>
        <div className="profile-grid">
          {AXES.map((axis) => {
            const score = phone.aspects.find((a) => a.aspect === axis);
            if (!score) {
              return (
                <div className="profile-cell profile-cell--missing" key={axis}>
                  <span className="profile-cell__name">{LABELS[axis]}</span>
                  <span className="profile-cell__value">&mdash;</span>
                  <span className="profile-cell__note">too few mentions to score</span>
                </div>
              );
            }
            return (
              <div className="profile-cell" key={axis}>
                <span className="profile-cell__name">{LABELS[axis]}</span>
                <span className={`profile-cell__value ${scoreTone(score.score)}`}>
                  {score.score.toFixed(1)}
                </span>
                <span className="profile-cell__note">
                  {score.source === "listed_price"
                    ? "from the listed price"
                    : `${score.mentions.toLocaleString()} mentions`}
                </span>
              </div>
            );
          })}
        </div>
        <p className="muted small">
          Scores run 1&ndash;10. The Price axis comes from the listed price, not from
          review sentiment: price opinions in the training data are 85.6% positive,
          so a sentiment-driven price score cannot tell phones apart.
        </p>
      </section>

      <section className="card" aria-labelledby="evidence-heading">
        <div className="card__head">
          <h2 id="evidence-heading">Why these scores</h2>
        </div>
        <p className="muted">
          Sentences the model picked out, strongest praise and strongest criticism
          for each aspect.
        </p>
        <EvidenceList items={phone.evidence} />
      </section>

      <ReviewForm modelKey={phone.model_key} />
    </div>
  );
}
