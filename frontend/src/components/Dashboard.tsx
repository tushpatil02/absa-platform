import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { BatchAnalyzeResponse, Polarity } from "../types";
import { ScoreMeter } from "./AspectCard";

/** Same validated diverging set as the EDA charts. Never red/green. */
const POLARITY_COLOUR: Record<Polarity, string> = {
  negative: "var(--polarity-negative)",
  neutral: "var(--polarity-neutral)",
  positive: "var(--polarity-positive)",
};

const POLARITY_ORDER: Polarity[] = ["negative", "neutral", "positive"];

function Legend() {
  return (
    <div className="legend">
      {POLARITY_ORDER.map((polarity) => (
        <span className="legend__item" key={polarity}>
          <span
            className="legend__swatch"
            style={{ background: POLARITY_COLOUR[polarity] }}
            aria-hidden="true"
          />
          {polarity.charAt(0).toUpperCase() + polarity.slice(1)}
        </span>
      ))}
    </div>
  );
}

interface TooltipPayload {
  payload: { name: string; mentions: number; negative: number; neutral: number; positive: number };
}

function ShareTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayload[] }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div
      style={{
        background: "var(--surface-card)",
        border: "1px solid var(--line-strong)",
        borderRadius: 8,
        padding: "9px 11px",
        fontSize: 12,
        boxShadow: "var(--shadow-card)",
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{row.name}</div>
      <div style={{ color: "var(--ink-secondary)" }}>
        {row.mentions} mention{row.mentions === 1 ? "" : "s"}
      </div>
      {POLARITY_ORDER.map((polarity) => (
        <div key={polarity} style={{ color: "var(--ink-secondary)" }}>
          {polarity}: {row[polarity].toFixed(0)}%
        </div>
      ))}
    </div>
  );
}

export function Dashboard({ data }: { data: BatchAnalyzeResponse }) {
  const chartData = data.aspects.map((aspect) => ({
    name: aspect.display_name,
    mentions: aspect.mentions,
    score: aspect.average_score ?? 0,
    negative: (aspect.shares.negative ?? 0) * 100,
    neutral: (aspect.shares.neutral ?? 0) * 100,
    positive: (aspect.shares.positive ?? 0) * 100,
  }));

  // Sorted by share negative so the chart answers "what are people complaining
  // about?" rather than merely listing aspects.
  const byNegative = [...chartData].sort((a, b) => b.negative - a.negative);

  // Changes whenever the analysed set changes, forcing a clean remount rather
  // than letting Recharts reuse geometry across a resize.
  const chartKey = `${data.reviews_analyzed}-${data.aspects.length}`;

  return (
    <>
      <div className="stats">
        <div className="stat">
          <div className="stat__label">Reviews analysed</div>
          <div className="stat__value">{data.reviews_analyzed.toLocaleString()}</div>
          {data.reviews_skipped > 0 && (
            <div className="stat__sub">{data.reviews_skipped} skipped as unusable</div>
          )}
        </div>
        <div className="stat">
          <div className="stat__label">Overall sentiment</div>
          <div className="stat__value">
            {data.overall_score !== null ? data.overall_score.toFixed(1) : "—"}
            <span style={{ fontSize: 14, color: "var(--ink-muted)", fontWeight: 400 }}> / 10</span>
          </div>
        </div>
        <div className="stat">
          <div className="stat__label">Most positive aspect</div>
          <div className="stat__value" style={{ fontSize: 19 }}>
            {data.most_positive?.display_name ?? "—"}
          </div>
          {data.most_positive?.average_score != null && (
            <div className="stat__sub">{data.most_positive.average_score.toFixed(1)} / 10</div>
          )}
        </div>
        <div className="stat">
          <div className="stat__label">Most negative aspect</div>
          <div className="stat__value" style={{ fontSize: 19 }}>
            {data.most_negative?.display_name ?? "—"}
          </div>
          {data.most_negative?.average_score != null && (
            <div className="stat__sub">{data.most_negative.average_score.toFixed(1)} / 10</div>
          )}
        </div>
      </div>

      <section className="card">
        <h2 className="card__title">Polarity mix by aspect</h2>
        <p className="card__hint">
          Share of reviews that mention each aspect, sorted by how negative they are. Percentages
          are of the reviews discussing that aspect, not of all reviews.
        </p>
        <Legend />
        <div className="chart-wrap">
          <ResponsiveContainer
            key={`polarity-${chartKey}`}
            width="100%"
            height={Math.max(220, byNegative.length * 34)}
          >
            <BarChart data={byNegative} layout="vertical" margin={{ left: 8, right: 16 }} barSize={17}>
              <XAxis
                type="number"
                domain={[0, 100]}
                unit="%"
                tick={{ fill: "var(--ink-muted)", fontSize: 11 }}
                axisLine={{ stroke: "var(--line-strong)" }}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={128}
                tick={{ fill: "var(--ink-secondary)", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<ShareTooltip />} cursor={{ fill: "var(--surface-sunken)" }} />
              {POLARITY_ORDER.map((polarity) => (
                <Bar
                  key={polarity}
                  dataKey={polarity}
                  stackId="polarity"
                  fill={POLARITY_COLOUR[polarity]}
                  // A hairline in the surface colour separates adjacent segments.
                  stroke="var(--surface-card)"
                  strokeWidth={1.5}
                  // Animation caches bar geometry across renders. ResponsiveContainer
                  // reports a small width on first paint and its real width a frame
                  // later; the axis re-scales but animated bars keep the stale scale,
                  // so every bar rendered ~7x too short while the axis read 0-100%.
                  isAnimationActive={false}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="card">
        <h2 className="card__title">Most discussed aspects</h2>
        <p className="card__hint">How many of the analysed reviews mention each aspect.</p>
        <div className="chart-wrap">
          <ResponsiveContainer
            key={`mentions-${chartKey}`}
            width="100%"
            height={Math.max(200, chartData.length * 32)}
          >
            <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 28 }} barSize={16}>
              <XAxis
                type="number"
                tick={{ fill: "var(--ink-muted)", fontSize: 11 }}
                axisLine={{ stroke: "var(--line-strong)" }}
                tickLine={false}
                allowDecimals={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={128}
                tick={{ fill: "var(--ink-secondary)", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "var(--surface-sunken)" }}
                contentStyle={{
                  background: "var(--surface-card)",
                  border: "1px solid var(--line-strong)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
              <Bar dataKey="mentions" name="Mentions" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                {chartData.map((row) => (
                  // Colour follows the entity's sentiment, not its rank.
                  <Cell
                    key={row.name}
                    fill={
                      row.score >= 6.25
                        ? "var(--polarity-positive)"
                        : row.score <= 4.75
                          ? "var(--polarity-negative)"
                          : "var(--polarity-neutral)"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="card">
        <h2 className="card__title">Aspect breakdown</h2>
        <p className="card__hint">
          The same data as a table — every figure above is readable here without relying on colour.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Aspect</th>
                <th className="num">Mentions</th>
                <th className="num">Avg score</th>
                <th className="num">Positive</th>
                <th className="num">Neutral</th>
                <th className="num">Negative</th>
                <th style={{ minWidth: 150 }}>Sentiment</th>
              </tr>
            </thead>
            <tbody>
              {data.aspects.map((aspect) => (
                <tr key={aspect.aspect}>
                  <td>{aspect.display_name}</td>
                  <td className="num">{aspect.mentions}</td>
                  <td className="num">
                    {aspect.average_score !== null ? aspect.average_score.toFixed(1) : "—"}
                  </td>
                  <td className="num">{((aspect.shares.positive ?? 0) * 100).toFixed(0)}%</td>
                  <td className="num">{((aspect.shares.neutral ?? 0) * 100).toFixed(0)}%</td>
                  <td className="num">{((aspect.shares.negative ?? 0) * 100).toFixed(0)}%</td>
                  <td>
                    {aspect.average_score !== null && <ScoreMeter score={aspect.average_score} />}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
