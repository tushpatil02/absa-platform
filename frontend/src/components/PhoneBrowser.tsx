/**
 * The catalogue, browsable without setting any sliders.
 *
 * Useful on its own, and useful as a check on the recommender: if the ranking
 * ever looks arbitrary, this shows the same profiles unranked so the numbers
 * can be compared directly.
 */

import { useEffect, useState } from "react";
import { ApiError, api } from "../api/client";
import { SimulatedBadge } from "./Recommender";
import { AXES } from "../types";
import type { Axis, PhoneListResponse } from "../types";

const LABELS: Record<Axis, string> = {
  battery: "Battery",
  camera: "Camera",
  price: "Price",
  display: "Display",
  performance: "Processor",
};

const PAGE_SIZE = 24;

function scoreTone(score: number): string {
  if (score >= 7.5) return "score--positive";
  if (score >= 4.5) return "score--neutral";
  return "score--negative";
}

export function PhoneBrowser({ onOpen }: { onOpen: (modelKey: string) => void }) {
  const [query, setQuery] = useState("");
  const [brand, setBrand] = useState("");
  const [page, setPage] = useState(0);
  const [data, setData] = useState<PhoneListResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Typing filters the list, so wait for a pause before asking the server.
    const timer = window.setTimeout(() => {
      api
        .phones({
          q: query || undefined,
          brand: brand || undefined,
          limit: PAGE_SIZE,
          offset: page * PAGE_SIZE,
        })
        .then((response) => {
          setData(response);
          setError(null);
        })
        .catch((cause: unknown) =>
          setError(cause instanceof ApiError ? cause.message : "Could not load the catalogue."),
        );
    }, 250);
    return () => window.clearTimeout(timer);
  }, [query, brand, page]);

  if (error) {
    return (
      <div className="alert alert--error" role="alert">
        <span className="alert__icon">!</span>
        <span>{error}</span>
      </div>
    );
  }

  if (!data) {
    return (
      <p className="empty">
        <span className="spinner" /> Loading the catalogue…
      </p>
    );
  }

  const pages = Math.ceil(data.total / PAGE_SIZE);

  return (
    <div className="browser">
      <div className="card browser__filters">
        <input
          className="input"
          type="search"
          placeholder="Search by name…"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setPage(0);
          }}
          aria-label="Search phones"
        />
        <select
          className="input"
          value={brand}
          onChange={(event) => {
            setBrand(event.target.value);
            setPage(0);
          }}
          aria-label="Filter by brand"
        >
          <option value="">All brands</option>
          {data.brands.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <span className="muted">{data.total} phones</span>
      </div>

      {data.phones.length === 0 ? (
        <p className="empty">No phone matches that filter.</p>
      ) : (
        <ul className="phone-grid">
          {data.phones.map((phone) => (
            <li className="phone-tile" key={phone.model_key}>
              <button className="phone-tile__button" onClick={() => onOpen(phone.model_key)}>
                {phone.image ? (
                  <img
                    className="phone-tile__image"
                    src={phone.image}
                    alt=""
                    loading="lazy"
                    onError={(event) => {
                      event.currentTarget.style.display = "none";
                    }}
                  />
                ) : null}
                <span className="phone-tile__name">
                  {phone.name}
                  {phone.simulated && <SimulatedBadge />}
                </span>
                <span className="phone-tile__meta">
                  {phone.price != null && <span>${phone.price.toFixed(0)}</span>}
                  <span>{phone.reviews_total.toLocaleString()} reviews</span>
                </span>
                <span className="phone-tile__scores">
                  {AXES.map((axis) => {
                    const score = phone.aspects.find((a) => a.aspect === axis);
                    return (
                      <span className="phone-tile__score" key={axis} title={LABELS[axis]}>
                        <em>{LABELS[axis].slice(0, 4)}</em>
                        <strong className={score ? scoreTone(score.score) : ""}>
                          {score ? score.score.toFixed(1) : "—"}
                        </strong>
                      </span>
                    );
                  })}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}

      {pages > 1 && (
        <div className="pager">
          <button
            className="button button--ghost button--sm"
            disabled={page === 0}
            onClick={() => setPage((current) => current - 1)}
          >
            Previous
          </button>
          <span className="muted">
            Page {page + 1} of {pages}
          </span>
          <button
            className="button button--ghost button--sm"
            disabled={page + 1 >= pages}
            onClick={() => setPage((current) => current + 1)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
