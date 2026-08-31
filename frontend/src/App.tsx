import { Suspense, lazy, useEffect, useState } from "react";
import { api } from "./api/client";
import { Analyzer } from "./components/Analyzer";
import type { HealthResponse } from "./types";

// Recharts is ~60% of the bundle and only the dashboard needs it. The default
// tab is the single-review analyzer, so loading it lazily keeps the initial
// payload to what the first screen actually uses.
const ProductAnalyzer = lazy(() =>
  import("./components/ProductAnalyzer").then((m) => ({ default: m.ProductAnalyzer })),
);

type Tab = "single" | "product";

export default function App() {
  const [tab, setTab] = useState<Tab>("single");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    // localStorage can throw in private mode; a missing preference is fine.
    try {
      const stored = localStorage.getItem("absa-theme");
      if (stored === "light" || stored === "dark") return stored;
    } catch {
      /* ignore */
    }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("absa-theme", theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  useEffect(() => {
    // A failed health check is itself the signal the banner needs.
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <div className="app">
      <header className="site-header">
        <div className="container site-header__inner">
          <div className="brand">
            <span className="brand__name">ABSA Platform</span>
            <span className="brand__tag">Aspect-Based Sentiment Analysis</span>
          </div>
          <div className="header-tools">
            {health && (
              <span style={{ fontSize: 12, color: "var(--ink-muted)" }}>
                <span
                  className={`status-dot status-dot--${health.model_loaded ? "ok" : "bad"}`}
                  aria-hidden="true"
                />
                {health.model_loaded ? health.model : "no model"}
              </span>
            )}
            <button
              className="button button--ghost button--sm"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            >
              {theme === "dark" ? "Light" : "Dark"}
            </button>
          </div>
        </div>
      </header>

      <main>
        <div className="container">
          {health && !health.model_loaded && (
            <div className="alert alert--warn" role="alert" style={{ marginBottom: 20 }}>
              <span className="alert__icon">!</span>
              <span>
                The API is running but no model is loaded.
                {health.detail ? ` ${health.detail}` : " Run scripts/train_baseline.py, then restart the API."}
              </span>
            </div>
          )}

          <div className="tabs" role="tablist" aria-label="Analysis mode">
            <button
              className="tab"
              role="tab"
              aria-selected={tab === "single"}
              onClick={() => setTab("single")}
            >
              Single review
            </button>
            <button
              className="tab"
              role="tab"
              aria-selected={tab === "product"}
              onClick={() => setTab("product")}
            >
              Product dashboard
            </button>
          </div>

          {tab === "single" ? (
            <Analyzer />
          ) : (
            <Suspense
              fallback={
                <div className="card">
                  <p className="empty">
                    <span className="spinner" /> Loading dashboard…
                  </p>
                </div>
              }
            >
              <ProductAnalyzer />
            </Suspense>
          )}
        </div>
      </main>

      <footer className="site-footer">
        <div className="container">
          Scores are model probabilities mapped to a 1&ndash;10 scale
          (<code>1 + 9 &times; E[polarity]</code>) &mdash; see <code>docs/scoring.md</code>.
          Confidence is the model&rsquo;s certainty, not a guarantee of correctness.
        </div>
      </footer>
    </div>
  );
}
