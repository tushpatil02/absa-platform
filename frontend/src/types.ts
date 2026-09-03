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

// ---------------------------------------------------------------------------
// Catalogue and recommender — mirrors backend/app/schemas/phones.py
// ---------------------------------------------------------------------------

/** The five slider axes, in the order the UI lays them out. */
export const AXES = ["battery", "camera", "price", "display", "performance"] as const;
export type Axis = (typeof AXES)[number];

export interface AspectScore {
  aspect: string;
  display_name: string;
  /** 1–10. */
  score: number;
  mentions: number;
  /**
   * Where the number came from. `price` is derived from the listed price rather
   * than review sentiment — price opinions in the training data are 85.6%
   * positive, so a sentiment-driven Price axis cannot separate phones. The UI
   * must label this rather than implying shoppers praised the price.
   */
  source: "reviews" | "listed_price";
}

export interface PhoneSummary {
  model_key: string;
  name: string;
  brand: string;
  price: number | null;
  image: string | null;
  url: string | null;
  reviews_total: number;
  avg_rating: number | null;
  aspects: AspectScore[];
  /** False when an axis is missing; such phones are not recommended. */
  rankable: boolean;
}

export interface PhoneListResponse {
  phones: PhoneSummary[];
  total: number;
  limit: number;
  offset: number;
  brands: string[];
}

export interface Evidence {
  aspect: string;
  display_name: string;
  polarity: Polarity;
  score: number;
  sentence: string;
}

export interface PhoneDetail extends PhoneSummary {
  reviews_scored: number;
  evidence: Evidence[];
}

export interface Match {
  phone: PhoneSummary;
  /** Share of the requirement met, 0–100. */
  match_percent: number;
  shortfalls: Record<string, number>;
  worst_axis: string | null;
}

export interface RecommendResponse {
  matches: Match[];
  preferences: Record<string, number>;
  considered: number;
  /** Listed price the Price slider position corresponds to. */
  price_target: number | null;
}

export type Preferences = Record<Axis, number>;

export interface SubmitReviewResponse {
  review_id: number;
  phone: string;
  aspects: AspectScore[];
  evidence: Evidence[];
  overall_score: number | null;
  model: string;
}
