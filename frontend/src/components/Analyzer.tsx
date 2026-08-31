import { useState } from "react";
import { ApiError, api } from "../api/client";
import type { AnalyzeResponse } from "../types";
import { AspectCard } from "./AspectCard";

const MAX_CHARS = 5000;

/** Deliberately mixed — they show aspect-conditional sentiment, which is the point. */
const EXAMPLES = [
  "The display is beautiful and the camera takes excellent photos, but the battery life is disappointing.",
  "Fast delivery and great packaging. Sadly the phone itself is very slow and the software is buggy.",
  "Superb value for money. Build feels cheap though, and the speaker is quiet.",
];

export function Analyzer() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const tooLong = review.length > MAX_CHARS;
  const canSubmit = review.trim().length > 0 && !tooLong && !loading;

  async function submit(event?: React.FormEvent) {
    event?.preventDefault();
    if (!canSubmit) return;

    setLoading(true);
    setError(null);
    try {
      setResult(await api.analyze(review));
    } catch (caught) {
      setResult(null);
      setError(
        caught instanceof ApiError ? caught.message : "Something went wrong. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <form className="card" onSubmit={submit}>
        <h2 className="card__title">Analyse a review</h2>
        <p className="card__hint">
          Paste a customer review. The model detects which aspects it discusses and scores
          sentiment for each one separately.
        </p>

        <textarea
          className="textarea"
          value={review}
          onChange={(event) => setReview(event.target.value)}
          placeholder="e.g. The camera is excellent, but the battery drains very quickly."
          aria-label="Review text"
          aria-invalid={tooLong}
          // Ctrl/Cmd+Enter submits — Enter alone must still insert a newline.
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter") submit();
          }}
        />

        <div className="field-row">
          <span className={`char-count${tooLong ? " char-count--over" : ""}`}>
            {review.length.toLocaleString()} / {MAX_CHARS.toLocaleString()} characters
          </span>
          <div style={{ display: "flex", gap: 8 }}>
            {review && (
              <button
                type="button"
                className="button button--ghost"
                onClick={() => {
                  setReview("");
                  setResult(null);
                  setError(null);
                }}
              >
                Clear
              </button>
            )}
            <button className="button" type="submit" disabled={!canSubmit}>
              {loading ? (
                <>
                  <span className="spinner" /> Analysing…
                </>
              ) : (
                "Analyse"
              )}
            </button>
          </div>
        </div>

        <div className="examples">
          <span className="examples__label">Try:</span>
          {EXAMPLES.map((example, index) => (
            <button
              key={example}
              type="button"
              className="button button--ghost button--sm"
              onClick={() => {
                setReview(example);
                setResult(null);
                setError(null);
              }}
            >
              Example {index + 1}
            </button>
          ))}
        </div>

        {tooLong && (
          <div className="alert alert--error" role="alert">
            <span className="alert__icon">!</span>
            <span>
              Review is {(review.length - MAX_CHARS).toLocaleString()} characters over the limit.
              Shorten it, or use the Product tab to analyse many reviews at once.
            </span>
          </div>
        )}

        {error && (
          <div className="alert alert--error" role="alert">
            <span className="alert__icon">!</span>
            <span>{error}</span>
          </div>
        )}
      </form>

      {result && (
        <section className="card" aria-live="polite">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              flexWrap: "wrap",
              gap: 10,
              marginBottom: 14,
            }}
          >
            <h2 className="card__title" style={{ marginBottom: 0 }}>
              {result.aspects.length} aspect{result.aspects.length === 1 ? "" : "s"} detected
            </h2>
            {result.overall_score !== null && (
              <span style={{ fontSize: 13, color: "var(--ink-secondary)" }}>
                Overall <strong>{result.overall_score.toFixed(1)}</strong> / 10
              </span>
            )}
          </div>

          <ul className="aspect-list" style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {result.aspects.map((aspect) => (
              <AspectCard key={aspect.aspect} aspect={aspect} />
            ))}
          </ul>

          <p style={{ fontSize: 12, color: "var(--ink-muted)", marginBottom: 0, marginTop: 16 }}>
            Model: <code>{result.model}</code>. Confidence is the model's probability for the
            predicted class, not a guarantee of correctness.
          </p>
        </section>
      )}
    </>
  );
}
