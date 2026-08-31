/**
 * API types.
 *
 * These mirror `backend/app/schemas/absa.py`. If the backend schema changes,
 * this file must change with it — the OpenAPI document at /openapi.json is the
 * authority, and `npm run typecheck` will not catch a drift here on its own.
 */

export type Polarity = "negative" | "neutral" | "positive";

export interface AspectSentiment {
  aspect: string;
  display_name: string;
  polarity: Polarity;
  /** 1–10, derived from the probability distribution. See docs/scoring.md. */
  score: number;
  /** Band name for the score, e.g. "Slightly Positive". */
  label: string;
  /** Probability of the predicted polarity — deliberately distinct from `score`. */
  confidence: number;
  /** Probability that this aspect is discussed at all. */
  detection_confidence: number;
  probabilities: Record<Polarity, number>;
}

export interface AnalyzeResponse {
  review: string;
  aspects: AspectSentiment[];
  overall_score: number | null;
  model: string;
}

export interface AspectSummary {
  aspect: string;
  display_name: string;
  mentions: number;
  mention_share: number;
  average_score: number | null;
  counts: Record<Polarity, number>;
  /** Shares among reviews that MENTION this aspect, not among all reviews. */
  shares: Record<Polarity, number>;
}

export interface BatchAnalyzeResponse {
  product_name: string | null;
  reviews_analyzed: number;
  reviews_skipped: number;
  overall_score: number | null;
  aspects: AspectSummary[];
  most_positive: AspectSummary | null;
  most_negative: AspectSummary | null;
  model: string;
}

export interface AspectInfo {
  id: string;
  display_name: string;
  description: string;
}

export interface AspectsResponse {
  aspects: AspectInfo[];
  polarities: Polarity[];
  score_range: { min: number; max: number };
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  model_loaded: boolean;
  model: string | null;
  detail: string | null;
}
