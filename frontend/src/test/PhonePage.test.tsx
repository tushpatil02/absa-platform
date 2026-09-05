/**
 * Tests for the phone detail page.
 *
 * The claims worth pinning are about honesty rather than layout: that every
 * score shows the sentences behind it, that criticism appears alongside praise,
 * that an aspect with too little evidence shows a dash rather than a number,
 * and that submitting a review does not appear to move the published scores.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "../api/client";
import { PhonePage } from "../components/PhonePage";
import type { PhoneDetail, SubmitReviewResponse } from "../types";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, api: { ...actual.api, phone: vi.fn(), submitReview: vi.fn() } };
});

const PHONE: PhoneDetail = {
  model_key: "acme nova",
  name: "Acme Nova",
  brand: "Acme",
  price: 299,
  image: null,
  url: null,
  reviews_total: 512,
  reviews_scored: 400,
  avg_rating: 4.1,
  rankable: true,
  simulated: false,
  aspects: [
    { aspect: "battery", display_name: "Battery", score: 8.2, mentions: 96, source: "reviews" },
    { aspect: "camera", display_name: "Camera", score: 3.4, mentions: 61, source: "reviews" },
    { aspect: "price", display_name: "Price", score: 6.7, mentions: 512, source: "listed_price" },
    { aspect: "display", display_name: "Display", score: 7.1, mentions: 44, source: "reviews" },
    // `performance` deliberately absent: too few mentions to score.
  ],
  evidence: [
    {
      aspect: "battery",
      display_name: "Battery",
      polarity: "positive",
      score: 9.4,
      sentence: "The battery easily lasts a day and a half of heavy use.",
    },
    {
      aspect: "camera",
      display_name: "Camera",
      polarity: "negative",
      score: 1.8,
      sentence: "Photos come out grainy whenever the light is low.",
    },
  ],
};

const ANALYSIS: SubmitReviewResponse = {
  review_id: 1,
  phone: "Acme Nova",
  overall_score: 5.5,
  model: "aspect=tfidf-logreg · sentiment=distilbert-base-uncased",
  aspects: [
    { aspect: "battery", display_name: "Battery", score: 9.1, mentions: 1, source: "reviews" },
    { aspect: "camera", display_name: "Camera", score: 1.9, mentions: 1, source: "reviews" },
  ],
  evidence: [
    {
      aspect: "battery",
      display_name: "Battery",
      polarity: "positive",
      score: 9.1,
      sentence: "The battery lasts all day.",
    },
    {
      aspect: "camera",
      display_name: "Camera",
      polarity: "negative",
      score: 1.9,
      sentence: "The camera is terrible in low light.",
    },
  ],
};

const getPhone = vi.mocked(api.phone);
const submitReview = vi.mocked(api.submitReview);

beforeEach(() => {
  getPhone.mockReset();
  submitReview.mockReset();
  getPhone.mockResolvedValue(PHONE);
  submitReview.mockResolvedValue(ANALYSIS);
});

describe("PhonePage", () => {
  it("shows the phone and how much of it was analysed", async () => {
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    expect(await screen.findByRole("heading", { name: "Acme Nova" })).toBeInTheDocument();
    expect(screen.getByText(/400 of 512 reviews analysed/)).toBeInTheDocument();
  });

  it("shows a score for every aspect that has one", async () => {
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    expect(await screen.findByText("8.2")).toBeInTheDocument();
    expect(screen.getByText("3.4")).toBeInTheDocument();
  });

  it("shows a dash where there was too little evidence, not a number", async () => {
    // Imputing a score would present a guess as a measurement.
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    await screen.findByText("8.2");
    expect(screen.getByText(/too few mentions to score/)).toBeInTheDocument();
  });

  it("says the Price score comes from the listed price", async () => {
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    // Said twice on purpose: once under the Price cell, once in the note that
    // explains why. getAllByText, not getByText.
    const mentions = await screen.findAllByText(/listed price/);
    expect(mentions.length).toBeGreaterThanOrEqual(2);
  });

  it("reports mention counts so a score's weight is visible", async () => {
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    expect(await screen.findByText("96 mentions")).toBeInTheDocument();
  });

  it("quotes the sentences behind the scores", async () => {
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    expect(await screen.findByText(/lasts a day and a half/)).toBeInTheDocument();
    expect(screen.getByText(/grainy whenever the light is low/)).toBeInTheDocument();
  });

  it("shows criticism alongside praise", async () => {
    // Quoting only the best sentences would make the page an advertisement.
    const { container } = render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    await screen.findByText(/lasts a day and a half/);
    expect(container.querySelector(".evidence__item--positive")).toBeTruthy();
    expect(container.querySelector(".evidence__item--negative")).toBeTruthy();
  });

  it("analyses a submitted review sentence by sentence", async () => {
    const user = userEvent.setup();
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    await screen.findByRole("heading", { name: "Acme Nova" });

    await user.type(
      screen.getByLabelText("Your review"),
      "The battery lasts all day. The camera is terrible in low light.",
    );
    await user.click(screen.getByRole("button", { name: /analyse my review/i }));

    expect(await screen.findByText("What the model found")).toBeInTheDocument();
    expect(screen.getByText(/The battery lasts all day\./)).toBeInTheDocument();
    expect(screen.getByText(/The camera is terrible in low light\./)).toBeInTheDocument();
  });

  it("keeps opposite opinions apart in the submitted analysis", async () => {
    // The whole point of sentence-level inference: one review, two verdicts.
    const user = userEvent.setup();
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    await screen.findByRole("heading", { name: "Acme Nova" });

    await user.type(screen.getByLabelText("Your review"), "Good battery. Bad camera.");
    await user.click(screen.getByRole("button", { name: /analyse my review/i }));

    const submitted = await screen.findByText("What the model found");
    const region = submitted.parentElement as HTMLElement;
    expect(within(region).getByText(/Battery 9\.1/)).toBeInTheDocument();
    expect(within(region).getByText(/Camera 1\.9/)).toBeInTheDocument();
  });

  it("tells the reader a submitted review does not change the scores", async () => {
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    // The sentence is broken by a <strong>not</strong>, so match the
    // contiguous run rather than the whole phrase.
    expect(await screen.findByText(/change the scores above/i)).toBeInTheDocument();
  });

  it("does not refetch the phone after a submission", async () => {
    // Refetching would imply the scores might have moved.
    const user = userEvent.setup();
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    await screen.findByRole("heading", { name: "Acme Nova" });
    expect(getPhone).toHaveBeenCalledTimes(1);

    await user.type(screen.getByLabelText("Your review"), "Great camera.");
    await user.click(screen.getByRole("button", { name: /analyse my review/i }));
    await screen.findByText("What the model found");

    expect(getPhone).toHaveBeenCalledTimes(1);
  });

  it("will not submit an empty review", async () => {
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    await screen.findByRole("heading", { name: "Acme Nova" });
    expect(screen.getByRole("button", { name: /analyse my review/i })).toBeDisabled();
  });

  it("surfaces a submission error", async () => {
    const user = userEvent.setup();
    submitReview.mockRejectedValue(new ApiError("Review is too long.", 422));
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    await screen.findByRole("heading", { name: "Acme Nova" });

    await user.type(screen.getByLabelText("Your review"), "Some text.");
    await user.click(screen.getByRole("button", { name: /analyse my review/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Review is too long.");
  });

  it("surfaces a load error instead of an empty page", async () => {
    getPhone.mockRejectedValue(new ApiError("No phone 'nope'", 404));
    render(<PhonePage modelKey="nope" onBack={() => {}} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("No phone 'nope'");
  });

  it("goes back", async () => {
    const user = userEvent.setup();
    const onBack = vi.fn();
    render(<PhonePage modelKey="acme nova" onBack={onBack} />);
    await user.click(await screen.findByRole("button", { name: /back to results/i }));
    await waitFor(() => expect(onBack).toHaveBeenCalled());
  });

  it("warns prominently when the reviews were generated", async () => {
    getPhone.mockResolvedValue({ ...PHONE, simulated: true });
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    const note = await screen.findByRole("note");
    expect(note).toHaveTextContent(/generated, not collected/i);
    expect(note).toHaveTextContent(/excluded from the reliability checks/i);
  });

  it("shows no such warning for a real phone", async () => {
    render(<PhonePage modelKey="acme nova" onBack={() => {}} />);
    await screen.findByRole("heading", { name: "Acme Nova" });
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
  });
});
