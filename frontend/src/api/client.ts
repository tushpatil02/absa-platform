/**
 * Typed API client.
 *
 * Every failure path produces an `ApiError` carrying a message the UI can show
 * directly. The backend returns FastAPI's `{detail: ...}` shape, where `detail`
 * is a string for our own raised errors and an array of field errors for
 * Pydantic validation failures — both are normalised here so components never
 * have to branch on it.
 */

import type {
  AnalyzeResponse,
  AspectsResponse,
  BatchAnalyzeResponse,
  HealthResponse,
  PhoneDetail,
  PhoneListResponse,
  Preferences,
  RecommendResponse,
  SubmitReviewResponse,
} from "../types";

/** Relative by default so the Vite dev proxy and same-origin deploys both work. */
const BASE_URL = import.meta.env.VITE_API_URL ?? "";

export class ApiError extends Error {
  // Declared as a field rather than a constructor parameter property: the
  // project builds with `erasableSyntaxOnly`, which rejects the shorthand.
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

interface FieldError {
  loc?: (string | number)[];
  msg?: string;
}

function readDetail(body: unknown, status: number): string {
  if (typeof body === "object" && body !== null && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;

    if (typeof detail === "string") return detail;

    // Pydantic validation errors arrive as an array of field errors.
    if (Array.isArray(detail)) {
      const messages = (detail as FieldError[])
        .map((entry) => {
          const field = entry.loc?.filter((part) => part !== "body").join(".");
          return field ? `${field}: ${entry.msg}` : entry.msg;
        })
        .filter(Boolean);
      if (messages.length) return messages.join("; ");
    }
  }
  return `Request failed (HTTP ${status}).`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    // fetch only rejects on network-level failure, which almost always means
    // the API is not running — say that rather than "Failed to fetch".
    throw new ApiError(
      "Cannot reach the API. Is the backend running on port 8000?",
      0,
    );
  }

  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      /* Non-JSON error body; fall through to the generic message. */
    }
    throw new ApiError(readDetail(body, response.status), response.status);
  }

  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),

  aspects: () => request<AspectsResponse>("/api/aspects"),

  analyze: (review: string, topK?: number) =>
    request<AnalyzeResponse>("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ review, top_k: topK ?? null }),
    }),

  analyzeBatch: (reviews: string[], productName?: string) =>
    request<BatchAnalyzeResponse>("/api/analyze/batch", {
      method: "POST",
      body: JSON.stringify({
        reviews,
        product_name: productName || null,
      }),
    }),

  phones: (
    params: {
      q?: string;
      brand?: string;
      limit?: number;
      offset?: number;
      includeSimulated?: boolean;
    } = {},
  ) => {
    const query = new URLSearchParams();
    if (params.q) query.set("q", params.q);
    if (params.brand) query.set("brand", params.brand);
    if (params.limit != null) query.set("limit", String(params.limit));
    if (params.offset != null) query.set("offset", String(params.offset));
    if (params.includeSimulated === false) query.set("include_simulated", "false");
    const suffix = query.toString();
    return request<PhoneListResponse>(`/api/phones${suffix ? `?${suffix}` : ""}`);
  },

  phone: (modelKey: string) =>
    request<PhoneDetail>(`/api/phones/${encodeURIComponent(modelKey)}`),

  recommend: (preferences: Preferences, limit = 12, includeSimulated = true) =>
    request<RecommendResponse>("/api/recommend", {
      method: "POST",
      body: JSON.stringify({ ...preferences, limit, include_simulated: includeSimulated }),
    }),

  submitReview: (modelKey: string, text: string, rating?: number) =>
    request<SubmitReviewResponse>(
      `/api/phones/${encodeURIComponent(modelKey)}/reviews`,
      {
        method: "POST",
        body: JSON.stringify({ text, rating: rating ?? null }),
      },
    ),
};
