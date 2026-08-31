/**
 * Tests for the API client.
 *
 * The point of this layer is that components never have to branch on FastAPI's
 * two different `detail` shapes, and that a dead backend produces a message a
 * user can act on rather than "Failed to fetch".
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/client";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function jsonResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

describe("success", () => {
  it("posts the review and returns the parsed body", async () => {
    const payload = { review: "x", aspects: [], overall_score: null, model: "m" };
    fetchMock.mockResolvedValue(jsonResponse(payload));

    await expect(api.analyze("Great camera")).resolves.toEqual(payload);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/analyze");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ review: "Great camera", top_k: null });
  });

  it("passes top_k through", async () => {
    fetchMock.mockResolvedValue(jsonResponse({}));
    await api.analyze("x", 3);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).top_k).toBe(3);
  });

  it("sends null rather than an empty product name", async () => {
    fetchMock.mockResolvedValue(jsonResponse({}));
    await api.analyzeBatch(["a", "b"], "");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).product_name).toBeNull();
  });
});

describe("error normalisation", () => {
  it("surfaces a string detail", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: "Review is too long." }, 413));

    await expect(api.analyze("x")).rejects.toMatchObject({
      message: "Review is too long.",
      status: 413,
    });
  });

  it("flattens Pydantic's array of field errors into one line", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(
        {
          detail: [
            { loc: ["body", "review"], msg: "String should have at least 1 character" },
            { loc: ["body", "top_k"], msg: "Input should be less than or equal to 12" },
          ],
        },
        422,
      ),
    );

    await expect(api.analyze("")).rejects.toThrow(
      "review: String should have at least 1 character; top_k: Input should be less than or equal to 12",
    );
  });

  it("falls back to a status message when the body is not JSON", async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => {
        throw new Error("not json");
      },
    } as unknown as Response);

    await expect(api.analyze("x")).rejects.toThrow("Request failed (HTTP 502).");
  });

  it("turns a network failure into an actionable message", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));

    const error = await api.analyze("x").catch((caught) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(0);
    expect(error.message).toMatch(/is the backend running/i);
  });
});
