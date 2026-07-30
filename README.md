# Market Intelligence from SEC 8-K Filings: A RAG-Based System

This project builds a **RAG-based system** that reads SEC **8-K filings**, classifies each event, and measures the stock's reaction in the days after — surfacing cases where the market may not have fully **priced in** the news yet. The output is a **ranked watchlist** of the highest-potential opportunities.

## 🔍 Project Goals

- Ingest 8-K filings and daily market data at scale.
- Classify each filing event and measure the post-event price reaction.
- Detect delayed or incomplete reactions and rank them into an actionable watchlist.

## 📄 Files Included

- **`notebooks/`** — the `NB1.1 → NB4` pipeline: market-price ingestion, 8-K ingestion, event classification, market-reaction analysis, and watchlist output.
- **`src/`** — module scaffold (`edgar`, `prices`, `classifier`, `reactions`, `scoring`, `db`, `config`).
- **`requirements.txt`** — Python dependencies.

> Raw 8-K text, market data, and generated outputs are excluded from the repo.

## 🧪 Methods Used

- SEC EDGAR data ingestion and text extraction
- Market/price data via `yfinance`; abnormal returns vs the S&P 500
- RAG + rule-based event classification (LLM)
- Event-study analysis (post-event return windows)
- Scoring and ranking into a watchlist

## 📈 Key Findings

- A **RAG-based system** reads each 8-K filing, classifies the event, and measures the stock's reaction over the following days.
- Some filings show a **delayed or incomplete reaction** — signals the market hasn't fully "priced in" yet.
- The resulting watchlist prioritized opportunities with **~1.6x higher model-estimated upside** than an unranked baseline.

## 🛠 Technologies

- Python (pandas)
- SEC EDGAR, `yfinance`
- LLM / RAG (Anthropic API) for event classification
- Jupyter notebooks; event-study analysis

## ✅ Outcome

Analysts get a ranked watchlist of names where the market may not yet have priced in an 8-K event — focusing attention on the highest-potential opportunities.

---

By **Parul Chaudhary** · [LinkedIn](https://www.linkedin.com/in/parulchaud) · [Email](mailto:parul.jaswant@gmail.com)
