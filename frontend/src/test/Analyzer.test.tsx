/**
 * Tests for the single-review analyzer.
 *
 * The API is mocked at the client boundary rather than at `fetch`, so these
 * cover the component's own behaviour — submission, rendering, error states —
 * without re-testing the network layer (which `client.test.ts` covers).
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Analyzer } from "../components/Analyzer";
import { ApiError, api } from "../api/client";
import type { AnalyzeResponse } from "../types";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, api: { ...actual.api, analyze: vi.fn() } };
});

const MIXED: AnalyzeResponse = {
  review: "Great camera but poor battery.",
  overall_score: 5.4,
  model: "baseline:tfidf+logreg",
  aspects: [
    {
      aspect: "camera",
      display_name: "Camera",
      polarity: "positive",
      score: 9.2,
      label: "Positive",
      confidence: 0.85,
      detection_confidence: 0.98,
      probabilities: { negative: 0.04, neutral: 0.11, positive: 0.85 },
    },
    {
      aspect: "battery",
      display_name: "Battery",
      polarity: "negative",
      score: 2.1,
      label: "Negative",
      confidence: 0.79,
      detection_confidence: 0.94,
      probabilities: { negative: 0.79, neutral: 0.13, positive: 0.08 },
    },
  ],
};

const mockAnalyze = vi.mocked(api.analyze);

beforeEach(() => {
  mockAnalyze.mockReset();
});

describe("submission", () => {
  it("disables the button until there is text", async () => {
    render(<Analyzer />);
    const button = screen.getByRole("button", { name: /^analyse$/i });
    expect(button).toBeDisabled();

    await userEvent.type(screen.getByLabelText(/review text/i), "Great camera");
    expect(button).toBeEnabled();
  });

  it("does not submit whitespace only", async () => {
    render(<Analyzer />);
    await userEvent.type(screen.getByLabelText(/review text/i), "   ");
    expect(screen.getByRole("button", { name: /^analyse$/i })).toBeDisabled();
    expect(mockAnalyze).not.toHaveBeenCalled();
  });

  it("sends the review text to the API", async () => {
    mockAnalyze.mockResolvedValue(MIXED);
    render(<Analyzer />);

    await userEvent.type(screen.getByLabelText(/review text/i), "Great camera but poor battery.");
    await userEvent.click(screen.getByRole("button", { name: /^analyse$/i }));

    await waitFor(() =>
      expect(mockAnalyze).toHaveBeenCalledWith("Great camera but poor battery."),
    );
  });

  it("fills the textarea from an example button", async () => {
    render(<Analyzer />);
    const textarea = screen.getByLabelText(/review text/i) as HTMLTextAreaElement;
    expect(textarea.value).toBe("");

    await userEvent.click(screen.getByRole("button", { name: /example 1/i }));

    // The examples are deliberately mixed reviews, so each mentions several aspects.
    expect(textarea.value).toMatch(/battery/i);
    expect(screen.getByRole("button", { name: /^analyse$/i })).toBeEnabled();
  });
});

describe("result rendering", () => {
  it("renders every aspect with polarity, score and both confidences", async () => {
    mockAnalyze.mockResolvedValue(MIXED);
    render(<Analyzer />);

    await userEvent.type(screen.getByLabelText(/review text/i), "Great camera but poor battery.");
    await userEvent.click(screen.getByRole("button", { name: /^analyse$/i }));

    expect(await screen.findByText("Camera")).toBeInTheDocument();
    expect(screen.getByText("Battery")).toBeInTheDocument();
    expect(screen.getByText("9.2")).toBeInTheDocument();
    expect(screen.getByText("2.1")).toBeInTheDocument();

    // Score and confidence must be shown as distinct figures.
    expect(screen.getByText(/sentiment confidence 85%/i)).toBeInTheDocument();
    expect(screen.getByText(/aspect detected 98%/i)).toBeInTheDocument();
  });

  it("labels polarity in text, not by colour alone", async () => {
    mockAnalyze.mockResolvedValue(MIXED);
    const { container } = render(<Analyzer />);
    await userEvent.type(screen.getByLabelText(/review text/i), "x");
    await userEvent.click(screen.getByRole("button", { name: /^analyse$/i }));
    await screen.findByText("Camera");

    // Scoped to the pills: "Positive" also appears as the score band label, so
    // an unscoped query matches twice and tells us nothing about the pill.
    const pills = [...container.querySelectorAll(".pill")].map((p) => p.textContent);
    expect(pills).toEqual(["Positive", "Negative"]);
  });

  it("exposes the score meter to assistive tech", async () => {
    mockAnalyze.mockResolvedValue(MIXED);
    render(<Analyzer />);
    await userEvent.type(screen.getByLabelText(/review text/i), "x");
    await userEvent.click(screen.getByRole("button", { name: /^analyse$/i }));

    const meters = await screen.findAllByRole("meter");
    expect(meters).toHaveLength(2);
    expect(meters[0]).toHaveAttribute("aria-valuenow", "9.2");
    expect(meters[0]).toHaveAttribute("aria-valuemin", "1");
    expect(meters[0]).toHaveAttribute("aria-valuemax", "10");
  });

  it("shows the overall score and the model name", async () => {
    mockAnalyze.mockResolvedValue(MIXED);
    render(<Analyzer />);
    await userEvent.type(screen.getByLabelText(/review text/i), "x");
    await userEvent.click(screen.getByRole("button", { name: /^analyse$/i }));

    expect(await screen.findByText(/2 aspects detected/i)).toBeInTheDocument();
    expect(screen.getByText("5.4")).toBeInTheDocument();
    expect(screen.getByText("baseline:tfidf+logreg")).toBeInTheDocument();
  });
});

describe("error states", () => {
  it("shows the API's message rather than a generic one", async () => {
    mockAnalyze.mockRejectedValue(new ApiError("Review contains no usable text.", 422));
    render(<Analyzer />);

    await userEvent.type(screen.getByLabelText(/review text/i), "!!!");
    await userEvent.click(screen.getByRole("button", { name: /^analyse$/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Review contains no usable text.");
  });

  it("explains an unreachable API", async () => {
    mockAnalyze.mockRejectedValue(
      new ApiError("Cannot reach the API. Is the backend running on port 8000?", 0),
    );
    render(<Analyzer />);

    await userEvent.type(screen.getByLabelText(/review text/i), "Great camera");
    await userEvent.click(screen.getByRole("button", { name: /^analyse$/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/cannot reach the api/i);
  });

  it("falls back to a generic message for a non-ApiError", async () => {
    mockAnalyze.mockRejectedValue(new TypeError("boom"));
    render(<Analyzer />);

    await userEvent.type(screen.getByLabelText(/review text/i), "Great camera");
    await userEvent.click(screen.getByRole("button", { name: /^analyse$/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/something went wrong/i);
  });

  it("blocks submission past the character limit and says by how much", async () => {
    render(<Analyzer />);
    const textarea = screen.getByLabelText(/review text/i);

    // paste rather than type: 5001 keystrokes would be glacial
    await userEvent.click(textarea);
    await userEvent.paste("x".repeat(5001));

    expect(screen.getByRole("button", { name: /^analyse$/i })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent(/1 characters over the limit/i);
    expect(mockAnalyze).not.toHaveBeenCalled();
  });

  it("clears a previous error on a successful retry", async () => {
    mockAnalyze.mockRejectedValueOnce(new ApiError("Server exploded", 500));
    mockAnalyze.mockResolvedValueOnce(MIXED);
    render(<Analyzer />);

    await userEvent.type(screen.getByLabelText(/review text/i), "Great camera");
    await userEvent.click(screen.getByRole("button", { name: /^analyse$/i }));
    expect(await screen.findByRole("alert")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /^analyse$/i }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(screen.getByText("Camera")).toBeInTheDocument();
  });
});
