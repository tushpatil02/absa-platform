import { useRef, useState } from "react";
import { ApiError, api } from "../api/client";
import type { BatchAnalyzeResponse } from "../types";
import { Dashboard } from "./Dashboard";

const MAX_REVIEWS = 500;

const SAMPLE = `Camera is superb, photos look professional. Battery lasts all day too.
Battery dies after four hours. Really disappointing for the price.
Fast delivery and the packaging was excellent. Phone works great.
The screen is gorgeous but it is far too expensive for what you get.
Terrible customer service. Took three weeks to get a reply.
Great value for money. Does everything I need.
Software is buggy and it freezes constantly. Camera is decent though.
Build quality feels cheap, but the display is bright and sharp.`;

/**
 * Parses pasted text or an uploaded CSV into a list of reviews.
 *
 * For CSV, the first column is taken unless a header names a review-ish column.
 * This is deliberately forgiving: a portfolio demo should accept the messy CSV a
 * user actually has, not demand a schema.
 */
function parseReviews(raw: string, isCsv: boolean): string[] {
  const lines = raw.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (!isCsv) return lines;

  const header = lines[0]?.toLowerCase() ?? "";
  const headerCells = splitCsvLine(header);
  const named = headerCells.findIndex((cell) =>
    ["review", "text", "review_text", "comment", "body", "content"].includes(cell.trim()),
  );
  const column = named >= 0 ? named : 0;
  const rows = named >= 0 ? lines.slice(1) : lines;

  return rows
    .map((line) => (splitCsvLine(line)[column] ?? "").trim().replace(/^"|"$/g, ""))
    .filter(Boolean);
}

/** Minimal CSV splitter that respects double-quoted fields containing commas. */
function splitCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (inQuotes && line[index + 1] === '"') {
        current += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (character === "," && !inQuotes) {
      cells.push(current);
      current = "";
    } else {
      current += character;
    }
  }
  cells.push(current);
  return cells;
}

export function ProductAnalyzer() {
  const [text, setText] = useState("");
  const [productName, setProductName] = useState("");
  const [data, setData] = useState<BatchAnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const reviews = parseReviews(text, false);
  const overLimit = reviews.length > MAX_REVIEWS;
  const canSubmit = reviews.length > 0 && !overLimit && !loading;

  async function handleFile(file: File) {
    setError(null);
    try {
      const raw = await file.text();
      const parsed = parseReviews(raw, file.name.toLowerCase().endsWith(".csv"));
      if (!parsed.length) {
        setError("No reviews found in that file.");
        return;
      }
      setText(parsed.join("\n"));
      if (!productName) setProductName(file.name.replace(/\.(csv|txt)$/i, ""));
    } catch {
      setError("Could not read that file.");
    }
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;

    setLoading(true);
    setError(null);
    try {
      setData(await api.analyzeBatch(reviews, productName));
    } catch (caught) {
      setData(null);
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
        <h2 className="card__title">Analyse a product's reviews</h2>
        <p className="card__hint">
          One review per line, or upload a .txt / .csv file. This answers what a star rating
          cannot: which aspects customers actually like and dislike.
        </p>

        <input
          className="textarea"
          style={{ minHeight: 0, marginBottom: 10 }}
          value={productName}
          onChange={(event) => setProductName(event.target.value)}
          placeholder="Product name (optional)"
          aria-label="Product name"
          maxLength={200}
        />

        <textarea
          className="textarea"
          style={{ minHeight: 190 }}
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Paste reviews, one per line…"
          aria-label="Reviews, one per line"
          aria-invalid={overLimit}
        />

        <div className="field-row">
          <span className={`char-count${overLimit ? " char-count--over" : ""}`}>
            {reviews.length.toLocaleString()} review{reviews.length === 1 ? "" : "s"}
            {overLimit && ` — limit is ${MAX_REVIEWS}`}
          </span>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <input
              ref={fileInput}
              type="file"
              accept=".txt,.csv"
              hidden
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void handleFile(file);
                event.target.value = "";
              }}
            />
            <button
              type="button"
              className="button button--ghost"
              onClick={() => fileInput.current?.click()}
            >
              Upload file
            </button>
            <button
              type="button"
              className="button button--ghost"
              onClick={() => {
                setText(SAMPLE);
                setProductName("Sample Phone");
                setData(null);
              }}
            >
              Load sample
            </button>
            <button className="button" type="submit" disabled={!canSubmit}>
              {loading ? (
                <>
                  <span className="spinner" /> Analysing {reviews.length}…
                </>
              ) : (
                "Analyse reviews"
              )}
            </button>
          </div>
        </div>

        {overLimit && (
          <div className="alert alert--error" role="alert">
            <span className="alert__icon">!</span>
            <span>
              {reviews.length.toLocaleString()} reviews exceeds the {MAX_REVIEWS} limit per
              request. Split the file into smaller batches.
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

      {data ? (
        <div aria-live="polite">
          {data.product_name && (
            <h2 style={{ fontSize: 18, margin: "24px 0 12px" }}>{data.product_name}</h2>
          )}
          <Dashboard data={data} />
        </div>
      ) : (
        !loading && (
          <div className="card">
            <p className="empty">
              Paste reviews above, or load the sample, to see the aspect breakdown.
            </p>
          </div>
        )
      )}
    </>
  );
}
