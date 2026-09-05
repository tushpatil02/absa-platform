/**
 * Tests for the slider recommender.
 *
 * The interesting claims are the ones the interface makes to the user: that a
 * slider is a minimum rather than a weight, that the Price axis is labelled as
 * coming from the listed price, and that a phone which misses a requirement is
 * shown as missing it. Ranking arithmetic lives server-side and is covered by
 * tests/test_recommender.py.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "../api/client";
import { Recommender } from "../components/Recommender";
import type { AspectScore, Match, RecommendResponse } from "../types";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, api: { ...actual.api, recommend: vi.fn() } };
});

function aspects(values: Record<string, number>): AspectScore[] {
  return Object.entries(values).map(([aspect, score]) => ({
    aspect,
    display_name: aspect,
    score,
    mentions: 40,
    source: aspect === "price" ? "listed_price" : "reviews",
  }));
}

function match(name: string, percent: number, shortfalls: Record<string, number>): Match {
  return {
    phone: {
      model_key: name.toLowerCase(),
      name,
      brand: "Acme",
      price: 299,
      image: null,
      url: null,
      reviews_total: 512,
      avg_rating: 4.1,
      rankable: true,
      simulated: name.includes("Sim"),
      aspects: aspects({
        battery: 8.1,
        camera: 6.4,
        price: 5.5,
        display: 7.2,
        performance: 7.8,
      }),
    },
    match_percent: percent,
    shortfalls,
    worst_axis: Object.keys(shortfalls).find((axis) => shortfalls[axis] > 0) ?? null,
  };
}

const RESPONSE: RecommendResponse = {
  matches: [
    match("Acme Nova", 100, { battery: 0, camera: 0, price: 0, display: 0, performance: 0 }),
    match("Acme Prime", 72.5, { battery: 0, camera: 2.6, price: 0, display: 0, performance: 0 }),
  ],
  preferences: { battery: 5, camera: 5, price: 5, display: 5, performance: 5 },
  considered: 137,
  simulated_considered: 0,
  price_target: 261.4,
};

const recommend = vi.mocked(api.recommend);

beforeEach(() => {
  recommend.mockReset();
  recommend.mockResolvedValue(RESPONSE);
});

describe("Recommender", () => {
  it("requests matches on mount", async () => {
    render(<Recommender onOpen={() => {}} />);
    await waitFor(() => expect(recommend).toHaveBeenCalled());
    expect(await screen.findByText("Acme Nova")).toBeInTheDocument();
  });

  it("shows all five sliders", async () => {
    render(<Recommender onOpen={() => {}} />);
    for (const label of ["Battery", "Camera", "Price", "Display", "Processor"]) {
      expect(screen.getByRole("slider", { name: new RegExp(label) })).toBeInTheDocument();
    }
  });

  it("explains that a slider is a minimum, not a weight", async () => {
    render(<Recommender onOpen={() => {}} />);
    expect(screen.getByText(/minimum/i)).toBeInTheDocument();
    expect(screen.getByText(/never penalised for being better/i)).toBeInTheDocument();
  });

  it("labels the Price axis as coming from the listed price", async () => {
    // Price sentiment is 85.6% positive and cannot rank phones, so the UI must
    // not imply this score came from reviews.
    render(<Recommender onOpen={() => {}} />);
    expect(screen.getByText(/listed price/i)).toBeInTheDocument();
  });

  it("reads a slider position in words as well as digits", async () => {
    render(<Recommender onOpen={() => {}} />);
    // The default of 5 should read as something a shopper understands.
    expect(screen.getAllByText("decent").length).toBeGreaterThan(0);
  });

  it("marks a slider at the minimum as 'any'", async () => {
    render(<Recommender onOpen={() => {}} />);
    // fireEvent.change, not keyboard input: jsdom range inputs do not move in
    // response to arrow keys, so a keyboard-driven test silently asserts
    // nothing.
    fireEvent.change(screen.getByRole("slider", { name: /Camera/ }), {
      target: { value: "1" },
    });
    expect(await screen.findByText("any")).toBeInTheDocument();
  });

  it("re-requests when a slider moves", async () => {
    render(<Recommender onOpen={() => {}} />);
    await waitFor(() => expect(recommend).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByRole("slider", { name: /Battery/ }), {
      target: { value: "9" },
    });

    await waitFor(() => expect(recommend).toHaveBeenCalledTimes(2));
    expect(recommend.mock.calls[1][0].battery).toBe(9);
  });

  it("shows the match percentage for each result", async () => {
    render(<Recommender onOpen={() => {}} />);
    expect(await screen.findByText("100%")).toBeInTheDocument();
    expect(screen.getByText("73%")).toBeInTheDocument();
  });

  it("names the axis a phone falls short on", async () => {
    render(<Recommender onOpen={() => {}} />);
    expect(await screen.findByText(/short on Camera by 2\.6/i)).toBeInTheDocument();
  });

  it("says so when a phone meets every requirement", async () => {
    render(<Recommender onOpen={() => {}} />);
    expect(await screen.findByText(/meets everything/i)).toBeInTheDocument();
  });

  it("reports how many phones were ranked", async () => {
    render(<Recommender onOpen={() => {}} />);
    expect(await screen.findByText(/ranked from 137 phones/i)).toBeInTheDocument();
  });

  it("warns when every phone matches, rather than implying a real ranking", async () => {
    // With all sliders at the minimum nothing was asked for, so a list ordered
    // 100%, 100%, 100% must not read as a meaningful ranking.
    recommend.mockResolvedValue({
      ...RESPONSE,
      matches: [
        match("A", 100, { battery: 0, camera: 0, price: 0, display: 0, performance: 0 }),
        match("B", 100, { battery: 0, camera: 0, price: 0, display: 0, performance: 0 }),
      ],
    });
    render(<Recommender onOpen={() => {}} />);
    expect(await screen.findByText(/Every phone meets these requirements/i)).toBeInTheDocument();
  });

  it("opens a phone when its name is clicked", async () => {
    const user = userEvent.setup();
    const onOpen = vi.fn();
    render(<Recommender onOpen={onOpen} />);
    await user.click(await screen.findByText("Acme Nova"));
    expect(onOpen).toHaveBeenCalledWith("acme nova");
  });

  it("resets the sliders", async () => {
    const user = userEvent.setup();
    render(<Recommender onOpen={() => {}} />);
    const slider = screen.getByRole("slider", { name: /Battery/ }) as HTMLInputElement;

    fireEvent.change(slider, { target: { value: "7" } });
    await waitFor(() => expect(slider.value).toBe("7"));

    await user.click(screen.getByRole("button", { name: /reset/i }));
    await waitFor(() => expect(slider.value).toBe("5"));
  });

  it("shows the API error rather than an empty list", async () => {
    recommend.mockRejectedValue(new ApiError("Catalogue not built.", 503));
    render(<Recommender onOpen={() => {}} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Catalogue not built.");
  });

  it("translates the Price slider into a rough price", async () => {
    render(<Recommender onOpen={() => {}} />);
    expect(await screen.findByText(/around \$261/)).toBeInTheDocument();
  });

  it("badges a simulated phone in the results", async () => {
    // These scores come from generated text. Showing them without saying so is
    // the one thing this feature was not allowed to do.
    recommend.mockResolvedValue({
      ...RESPONSE,
      simulated_considered: 1,
      matches: [
        match("Sim Phone 2025", 92, {
          battery: 0,
          camera: 0,
          price: 0,
          display: 0,
          performance: 0,
        }),
      ],
    });
    render(<Recommender onOpen={() => {}} />);
    expect(await screen.findAllByText("simulated")).not.toHaveLength(0);
  });

  it("says how many ranked phones were simulated", async () => {
    recommend.mockResolvedValue({ ...RESPONSE, simulated_considered: 3 });
    render(<Recommender onOpen={() => {}} />);
    expect(await screen.findByText(/of the 137 ranked phones are/i)).toBeInTheDocument();
  });

  it("does not badge a real phone", async () => {
    render(<Recommender onOpen={() => {}} />);
    await screen.findByText("Acme Nova");
    expect(screen.queryByText("simulated")).not.toBeInTheDocument();
  });

  it("can exclude simulated phones and re-requests without them", async () => {
    const user = userEvent.setup();
    recommend.mockResolvedValue({ ...RESPONSE, simulated_considered: 2 });
    render(<Recommender onOpen={() => {}} />);
    await waitFor(() => expect(recommend).toHaveBeenCalledTimes(1));
    expect(recommend.mock.calls[0][2]).toBe(true);

    await user.click(await screen.findByRole("button", { name: /only real phones/i }));

    await waitFor(() => expect(recommend).toHaveBeenCalledTimes(2));
    expect(recommend.mock.calls[1][2]).toBe(false);
  });
});
