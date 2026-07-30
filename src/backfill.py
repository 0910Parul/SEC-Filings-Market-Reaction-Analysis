"""
backfill.py — Historical 8-K batch processor for ML training data collection
Not Yet Priced In · Team 10

Runs the full pipeline (parse → A2 features → Claude API → market reaction)
over historical years (default 2020-2022) for all 50 tracked tickers.

Output structure:
  data/
    2020/
      filings_text/     ← raw .txt files
      parse_summary.csv
      parse_summary_clean_text.csv
      stage_a_features.csv
      stage_a3_llm.csv
      market_reactions.csv
      backfill_labelled.csv   ← final ML training rows (all columns + label)
    2021/ ...
    2022/ ...
  backfill_master.csv   ← all years merged, ready for ML training

Usage:
  export ANTHROPIC_API_KEY=your_key
  python backfill.py --years 2020 2021 2022
  python backfill.py --years 2022          # single year
  python backfill.py --years 2022 --skip-llm  # features only, no API cost
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── Add repo root to path so src and parse_filing are importable ──────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config, backend
sys.path.insert(0, str(config.NOTEBOOKS_DIR))
from parse_filing import parse_filing, get_primary_item

# Use centralized logger if possible, otherwise keep local basicConfig
log = config.setup_logging("backfill")

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR      = config.DATA_DIR
CALIBRATION_CSV = DATA_DIR / "calibration_table.csv"

HEADERS = {"User-Agent": "Team10 MSBA contact@umn.edu", "Accept-Encoding": "gzip, deflate"}

SLEEP_EDGAR   = 0.2   # seconds between EDGAR requests (rate limit)
SLEEP_LLM     = 1.2   # seconds between Claude API calls
CHECKPOINT_N  = 25    # save checkpoint every N LLM calls

BENCHMARK     = "^GSPC"
MARKET_CLOSE  = 16    # 4 PM ET

# ── 50 tickers → CIK mapping ───────────────────────────────────────────────
TICKERS_TO_CIK = {
    "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044",
    "AMZN": "0001018724", "META": "0001326801", "NVDA": "0001045810",
    "ADBE": "0000796343", "CRM":  "0001108524", "JPM":  "0000019617",
    "BAC":  "0000070858", "GS":   "0000886982", "MS":   "0000895421",
    "WFC":  "0000072971", "BLK":  "0001364742", "JNJ":  "0000200406",
    "PFE":  "0000078003", "MRK":  "0000310158", "ABBV": "0001551152",
    "UNH":  "0000731766", "LLY":  "0000059478", "WMT":  "0000104169",
    "COST": "0000909832", "KO":   "0000021344", "PEP":  "0000077476",
    "NKE":  "0000320187", "SBUX": "0000829224", "BA":   "0000012927",
    "GE":   "0000040987", "HON":  "0000773840", "CAT":  "0000018230",
    "UNP":  "0000100885", "XOM":  "0000034088", "CVX":  "0000093410",
    "COP":  "0001163165", "SLB":  "0000087347", "DIS":  "0001001039",
    "NFLX": "0001065280", "CMCSA":"0001166691", "T":    "0000732717",
    "VZ":   "0000732712", "MMM":  "0000066740", "DOW":  "0001751788",
    "DD":   "0001666700", "FCX":  "0000831259", "TSLA": "0001318605",
    "INTC": "0000050863", "ORCL": "0001339439", "QCOM": "0000804328",
    "AXP":  "0000004962", "CVS":  "0000064803",
}

BROAD_CATEGORY_MAP = {
    "2.02": "Earnings",
    "1.01": "Material Agreement", "1.02": "Material Agreement", "1.03": "Material Agreement",
    "5.02": "Executive Change",
    "5.07": "Voting Results",
    "5.03": "Governance", "5.05": "Governance", "5.08": "Governance",
    "5.01": "Governance", "5.04": "Governance", "5.06": "Governance",
    "7.01": "Regulatory", "8.01": "Regulatory",
    "2.03": "Financial Obligation", "2.04": "Financial Obligation",
    "2.05": "Financial Obligation", "2.06": "Financial Obligation",
    "2.01": "Acquisition/Disposition",
    "3.01": "Securities", "3.02": "Securities", "3.03": "Securities",
    "4.01": "Accounting", "4.02": "Accounting",
}

FORWARD_TERMS = backend.FORWARD_TERMS if hasattr(backend, 'FORWARD_TERMS') else {
    "will","expect","expects","expected","anticipate","anticipates",
    "forecast","forecasts","project","projects","projected","projecting",
    "estimate","estimates","estimated","intend","intends","plan","plans",
    "planned","believe","believes","outlook","guidance","target","targets",
    "goal","goals","upcoming","future","forward","should","could","may",
    "might","would","pending","approximately",
}

LLM_SYSTEM_PROMPT = backend.SYSTEM_PROMPT

# ── Step 1: EDGAR — fetch filing list for one ticker + year ────────────────
def fetch_edgar_filings(ticker: str, cik: str, year: int) -> list[dict]:
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"EDGAR fetch failed for {ticker}: {e}")
        return []

    company = data.get("name", ticker)
    recent  = data.get("filings", {}).get("recent", {})
    forms   = recent.get("form", [])
    dates   = recent.get("filingDate", [])
    accessions   = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    year_str = str(year)
    results  = []
    for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
        if form != "8-K":
            continue
        if not date.startswith(year_str):
            continue
        acc_clean = acc.replace("-", "")
        cik_int   = int(cik)
        doc_url   = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{doc}"
        results.append({
            "ticker":    ticker,
            "cik":       cik,
            "company":   company,
            "accession": acc,
            "filed_date":date,
            "doc_url":   doc_url,
        })

    # EDGAR recent only goes back ~1000 filings; for older data use full-index
    # If we got no results for old years, fall back to full-text index
    if not results:
        results = fetch_edgar_full_index(ticker, cik, company, year)

    return results


def fetch_edgar_full_index(ticker: str, cik: str, company: str, year: int) -> list[dict]:
    """
    Fallback: use EDGAR quarterly full-text search index for years
    not covered by the submissions API recent window.
    """
    results = []
    cik_int = int(cik)
    for qtr in [1, 2, 3, 4]:
        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/company.idx"
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            for line in r.text.splitlines():
                if "8-K" not in line:
                    continue
                parts = line.split("|")
                if len(parts) < 5:
                    continue
                if str(cik_int) not in parts[0] and cik not in parts[0]:
                    continue
                acc_raw = parts[4].strip()
                acc_no  = acc_raw.replace("/", "").replace("-", "")
                date    = parts[3].strip()
                # Build URL to filing index
                idx_url = f"https://www.sec.gov/Archives/edgar/{acc_raw}-index.htm"
                results.append({
                    "ticker":    ticker,
                    "cik":       cik,
                    "company":   company,
                    "accession": acc_no[:10] + "-" + acc_no[10:12] + "-" + acc_no[12:],
                    "filed_date":date,
                    "doc_url":   idx_url,  # will be resolved below
                })
            time.sleep(SLEEP_EDGAR)
        except Exception as e:
            log.debug(f"Full index fetch failed {ticker} {year} Q{qtr}: {e}")
    return results


# ── Step 2: Download raw filing text ───────────────────────────────────────
def download_filing_text(doc_url: str, accession: str, ticker: str) -> str:
    try:
        r = requests.get(doc_url, headers=HEADERS, timeout=25)
        r.raise_for_status()
        text = r.text
        # Strip HTML tags if present
        if "<html" in text.lower() or "<body" in text.lower():
            from html.parser import HTMLParser
            class MLStripper(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.reset()
                    self.fed = []
                def handle_data(self, d): self.fed.append(d)
                def get_data(self): return " ".join(self.fed)
            s = MLStripper()
            s.feed(text)
            text = s.get_data()
        return text[:50000]  # cap at 50k chars
    except Exception as e:
        log.warning(f"Download failed {ticker} {accession}: {e}")
        return ""


# ── Step 3: A2 text features ───────────────────────────────────────────────
def compute_text_features(clean_text: str) -> dict:
    if not clean_text or not isinstance(clean_text, str):
        return {
            "numeric_density": 0.0,
            "forward_looking_density": 0.0,
            "financial_symbol_density": 0.0,
            "baseline_importance": 0.0,
        }
    words = clean_text.split()
    if not words:
        return {
            "numeric_density": 0.0,
            "forward_looking_density": 0.0,
            "financial_symbol_density": 0.0,
            "baseline_importance": 0.0,
        }
    n = len(words)
    nd  = sum(1 for w in words if any(c.isdigit() for c in w)) / n * 100
    fld = sum(
        1 for w in words
        if w.strip(".,;:!?\"'()[]").lower() in FORWARD_TERMS
    ) / n * 100
    fsd = (clean_text.count("$") + clean_text.count("%")) / n * 100
    bi  = min(1.0, (nd / 15 * 0.33 + fld / 8 * 0.33 + fsd / 5 * 0.34))
    return {
        "numeric_density":          round(nd,  4),
        "forward_looking_density":  round(fld, 4),
        "financial_symbol_density": round(fsd, 4),
        "baseline_importance":      round(bi,  4),
    }


# ── Step 4: Claude API importance scoring ──────────────────────────────────
def llm_score(clean_text: str, max_chars: int = 15000) -> dict:
    from openai import OpenAI
    try:
        if not config.OPENAI_API_KEY:
            return _llm_error("Missing OpenAI API Key")
            
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        truncated = clean_text[:max_chars]
        
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this 8-K filing:\n\n{truncated}"}
            ],
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        raw = content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
        r   = json.loads(raw)
        return {
            "llm_event_category": r.get("event_category", "Other"),
            "importance_score":   int(r.get("importance_score", 3)),
            "key_signals":        json.dumps(r.get("key_signals", [])),
            "reasoning":          r.get("reasoning", ""),
            "llm_confidence":     float(r.get("confidence", 0.5)),
            "grounding_rate":     _check_grounding(clean_text, r.get("key_signals", [])),
            "llm_error":          None,
        }
    except json.JSONDecodeError as e:
        return _llm_error(f"json_error: {e}")
    except Exception as e:
        return _llm_error(f"api_error: {str(e)[:80]}")


def _llm_error(reason: str) -> dict:
    return {
        "llm_event_category": "Other",
        "importance_score":   3,
        "key_signals":        "[]",
        "reasoning":          reason,
        "llm_confidence":     0.0,
        "grounding_rate":     0.0,
        "llm_error":          reason,
    }


def _check_grounding(clean_text: str, key_signals: list) -> float:
    if not key_signals:
        return 0.0
    norm = " ".join(str(clean_text).split())
    hits = [" ".join(str(s).split()) in norm for s in key_signals]
    return round(sum(hits) / len(hits), 3)


# ── Step 5: Market reaction ─────────────────────────────────────────────────
def get_event_day0(filing_date: str, calendar: pd.DatetimeIndex) -> pd.Timestamp | None:
    """
    Replicates NB1-1 get_event_day0 logic.
    For backfill we assume all historical filings were filed after market close
    (conservative assumption — most 8-Ks are). Day 0 = next trading day.
    """
    cal_dates = calendar.normalize()
    dt = pd.Timestamp(filing_date)
    # Next trading day on or after the filing date
    future = cal_dates[cal_dates >= dt]
    return future[0] if len(future) > 0 else None


def compute_event_windows(
    ticker: str,
    day0: pd.Timestamp,
    abnormal_returns: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> dict | None:
    """
    Exact replica of NB1-1 compute_event_windows().
    R_short  = CAR days 0-1
    R_medium = CAR days 0-3
    R_long   = CAR days 5-20
    """
    cal_dates = calendar.normalize()
    day0_norm = day0.normalize()

    if day0_norm not in cal_dates:
        return None

    idx = cal_dates.get_loc(day0_norm)

    def _car(start_offset: int, end_offset: int):
        s = idx + start_offset
        e = idx + end_offset + 1
        if e > len(cal_dates):
            return None
        window_dates = cal_dates[s:e]
        if ticker not in abnormal_returns.columns:
            return None
        vals = abnormal_returns.loc[window_dates, ticker].dropna()
        return float(vals.sum())

    r_short  = _car(0, 1)
    r_medium = _car(0, 3)
    r_long   = _car(5, 20)

    delayed = None
    if r_short is not None and r_long is not None:
        same_dir = (r_short * r_long) > 0
        delayed  = (abs(r_short) < 0.01) and (abs(r_long) > 0.03) and same_dir

    return {
        "R_short_0_1":  round(r_short,  5) if r_short  is not None else None,
        "R_medium_0_3": round(r_medium, 5) if r_medium is not None else None,
        "R_long_5_20":  round(r_long,   5) if r_long   is not None else None,
        "delayed_flag": delayed,
    }


def classify_reaction(r_short, r_long, short_thr, long_thr) -> str:
    """4-class label — same logic as NB3."""
    if r_short is None or r_long is None or pd.isna(r_short) or pd.isna(r_long):
        return "Unknown"
    same_dir = (r_short * r_long) > 0
    if abs(r_short) >= short_thr:
        return "Immediate"
    elif abs(r_short) < short_thr and abs(r_long) > long_thr and same_dir:
        return "Delayed"
    elif same_dir and abs(r_long) > 0:
        return "Gradual"
    else:
        return "No Reaction"


def build_price_data(year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """
    Download 1 year of price data for all 50 tickers + S&P 500.
    Returns (abnormal_returns DataFrame, prices DataFrame, trading calendar).
    """
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    fetch_start = f"{year - 1}-12-15"   # a little buffer for early-Jan events

    log.info(f"Downloading prices for {year} ({len(TICKERS_TO_CIK)} tickers + benchmark)...")
    all_tickers = list(TICKERS_TO_CIK.keys()) + [BENCHMARK]

    raw = yf.download(
        all_tickers,
        start=fetch_start,
        end=f"{year + 1}-01-15",
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices_all = raw["Close"]
    else:
        prices_all = raw[["Close"]]

    benchmark_ret = prices_all[BENCHMARK].pct_change().rename("benchmark")
    stock_prices  = prices_all[list(TICKERS_TO_CIK.keys())]
    stock_ret     = stock_prices.pct_change()
    abnormal_ret  = stock_ret.sub(benchmark_ret, axis=0)

    calendar = prices_all.loc[start:end].index

    log.info(f"  Prices: {stock_prices.shape} | Calendar: {len(calendar)} trading days")
    return abnormal_ret, stock_prices, calendar


# ── MOS_prospective from calibration table ─────────────────────────────────
def load_mos_assets() -> tuple[dict, dict]:
    if not CALIBRATION_CSV.exists():
        log.warning(f"calibration_table.csv not found — MOS_prospective will be None")
        return {}, {}

    calib = pd.read_csv(CALIBRATION_CSV)
    lift_lookup = {}
    for _, row in calib.iterrows():
        feat = row["feature"]
        val  = str(row["value"])
        lift = row["lift"] if pd.notna(row.get("lift")) else 1.0
        lift_lookup.setdefault(feat, {})[val] = lift

    # Fixed tier thresholds from 2023 corpus (approximate)
    tier_lo_hi = {
        "forward_looking_tier":     (1.2, 3.1),
        "numeric_density_tier":     (2.8, 7.4),
        "baseline_importance_tier": (0.38, 0.62),
        "llm_importance_tier":      (0.33, 0.67),
    }
    return lift_lookup, tier_lo_hi


def _get_tier(value, lo, hi) -> str:
    if pd.isna(value): return "mid"
    return "low" if value <= lo else "high" if value >= hi else "mid"


def score_mos_prospective(features: dict, lift_lookup: dict, tier_lo_hi: dict) -> float | None:
    if not lift_lookup:
        return None
    prior = 1.0
    cat = str(features.get("broad_category", "Other"))
    prior *= lift_lookup.get("broad_category", {}).get(cat, 1.0)

    imp_norm = (features.get("importance_score", 3) - 1) / 4
    tier = _get_tier(imp_norm, *tier_lo_hi["llm_importance_tier"])
    prior *= lift_lookup.get("llm_importance_tier", {}).get(tier, 1.0)

    tier = _get_tier(features.get("forward_looking_density", 0), *tier_lo_hi["forward_looking_tier"])
    prior *= lift_lookup.get("forward_looking_tier", {}).get(tier, 1.0)

    tier = _get_tier(features.get("numeric_density", 0), *tier_lo_hi["numeric_density_tier"])
    prior *= lift_lookup.get("numeric_density_tier", {}).get(tier, 1.0)

    tier = _get_tier(features.get("baseline_importance", 0.5), *tier_lo_hi["baseline_importance_tier"])
    prior *= lift_lookup.get("baseline_importance_tier", {}).get(tier, 1.0)

    # Normalise to 0-1 via sigmoid around 1.0
    return round(float(1 / (1 + np.exp(-2 * (prior - 1.0)))), 4)


# ── Main backfill runner ────────────────────────────────────────────────────
def run_backfill_year(year: int, skip_llm: bool = False):
    log.info(f"{'='*60}")
    log.info(f"  BACKFILL {year}")
    log.info(f"{'='*60}")

    # Set up year folder structure — files go directly into data/YEAR/
    year_dir    = DATA_DIR / str(year)
    text_dir    = year_dir
    text_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_csv = year_dir / "backfill_checkpoint.csv"
    output_csv     = year_dir / "backfill_labelled.csv"

    # Load already-processed accessions (resume support)
    done_accessions = set()
    if checkpoint_csv.exists():
        done_df = pd.read_csv(checkpoint_csv)
        done_accessions = set(done_df["accession"].tolist())
        log.info(f"Resuming — {len(done_accessions)} accessions already processed")

    # Load price data for the year
    abnormal_ret, stock_prices, calendar = build_price_data(year)

    # Quantile thresholds from this year's price data
    # (compute after we have reactions — use defaults for now, update below)
    short_thr = 0.01
    long_thr  = 0.03

    # Load MOS scoring assets
    lift_lookup, tier_lo_hi = load_mos_assets()

    # Collect all filings for the year
    log.info(f"Fetching EDGAR filing lists for {len(TICKERS_TO_CIK)} tickers...")
    all_filings = []
    for ticker, cik in TICKERS_TO_CIK.items():
        filings = fetch_edgar_filings(ticker, cik, year)
        all_filings.extend(filings)
        time.sleep(SLEEP_EDGAR)

    log.info(f"Found {len(all_filings)} 8-K filings for {year}")
    new_filings = [f for f in all_filings if f["accession"] not in done_accessions]
    log.info(f"  {len(new_filings)} new filings to process (after checkpoint skip)")

    # Process each filing
    rows = []
    llm_call_count = 0

    for i, filing in enumerate(new_filings):
        ticker   = filing["ticker"]
        acc      = filing["accession"]
        date_str = filing["filed_date"]

        log.info(f"[{i+1}/{len(new_filings)}] {ticker} {date_str} {acc}")

        # ── Download and parse ──────────────────────────────────────────────
        txt_path = text_dir / f"{ticker}_{acc.replace('-','_')}.txt"
        if txt_path.exists():
            raw_text = txt_path.read_text(encoding="utf-8", errors="replace")
        else:
            raw_text = download_filing_text(filing["doc_url"], acc, ticker)
            if raw_text:
                txt_path.write_text(raw_text, encoding="utf-8")
            time.sleep(SLEEP_EDGAR)

        if not raw_text:
            log.warning(f"  Skipping — no text downloaded")
            continue

        parsed  = parse_filing(raw_text)
        primary = get_primary_item(parsed)
        clean   = parsed.get("clean_text", "")

        if not clean or len(clean) < 50:
            log.info(f"  Skipping — clean_text too short ({len(clean)} chars)")
            continue

        primary_item = primary["number"] if primary else None
        broad_cat    = BROAD_CATEGORY_MAP.get(primary_item or "", "Other")

        # ── A2 text features ────────────────────────────────────────────────
        feats = compute_text_features(clean)

        # ── A3 LLM scoring ──────────────────────────────────────────────────
        if skip_llm:
            llm = {
                "llm_event_category": broad_cat,
                "importance_score":   3,
                "key_signals":        "[]",
                "reasoning":          "skipped",
                "llm_confidence":     0.0,
                "grounding_rate":     0.0,
                "llm_error":          "skip_llm=True",
            }
        else:
            llm = llm_score(clean)
            llm_call_count += 1
            time.sleep(SLEEP_LLM)

        # ── Market reaction ─────────────────────────────────────────────────
        day0 = get_event_day0(date_str, calendar)
        rxn  = None
        if day0 is not None:
            rxn = compute_event_windows(ticker, day0, abnormal_ret, calendar)

        r_short  = rxn["R_short_0_1"]  if rxn else None
        r_medium = rxn["R_medium_0_3"] if rxn else None
        r_long   = rxn["R_long_5_20"]  if rxn else None
        reaction_class = classify_reaction(r_short, r_long, short_thr, long_thr)

        # ── MOS_prospective ─────────────────────────────────────────────────
        mos_features = {
            "broad_category":          broad_cat,
            "importance_score":        llm["importance_score"],
            "forward_looking_density": feats["forward_looking_density"],
            "numeric_density":         feats["numeric_density"],
            "baseline_importance":     feats["baseline_importance"],
        }
        mos_p = score_mos_prospective(mos_features, lift_lookup, tier_lo_hi)

        # ── Assemble row ────────────────────────────────────────────────────
        row = {
            "year":         year,
            "ticker":       ticker,
            "company":      filing.get("company", ""),
            "accession":    acc,
            "filed_date":   date_str,
            "primary_item": primary_item,
            "broad_category": broad_cat,
            # A2
            "numeric_density":          feats["numeric_density"],
            "forward_looking_density":  feats["forward_looking_density"],
            "financial_symbol_density": feats["financial_symbol_density"],
            "baseline_importance":      feats["baseline_importance"],
            # A3
            "llm_event_category": llm["llm_event_category"],
            "importance_score":   llm["importance_score"],
            "key_signals":        llm["key_signals"],
            "reasoning":          llm["reasoning"],
            "llm_confidence":     llm["llm_confidence"],
            "grounding_rate":     llm["grounding_rate"],
            "llm_error":          llm["llm_error"],
            # Market reaction
            "R_short_0_1":  r_short,
            "R_medium_0_3": r_medium,
            "R_long_5_20":  r_long,
            "reaction_class": reaction_class,  # ← ground truth label
            # MOS
            "MOS_prospective": mos_p,
            # Parse quality
            "parse_confidence":  parsed["parse_confidence"],
            "clean_text_length": len(clean),
        }
        rows.append(row)

        # Checkpoint every N LLM calls
        if not skip_llm and llm_call_count % CHECKPOINT_N == 0 and llm_call_count > 0:
            ckpt = pd.DataFrame(rows)
            if checkpoint_csv.exists():
                ckpt = pd.concat([pd.read_csv(checkpoint_csv), ckpt], ignore_index=True)
            ckpt.to_csv(checkpoint_csv, index=False)
            log.info(f"  Checkpoint saved: {len(rows)} new rows")

    # ── Save year output ────────────────────────────────────────────────────
    if rows:
        new_df = pd.DataFrame(rows)

        # Update quantile thresholds from actual this-year data
        has_rxn = new_df["R_short_0_1"].notna() & new_df["R_long_5_20"].notna()
        if has_rxn.sum() > 10:
            short_thr_data = new_df.loc[has_rxn, "R_short_0_1"].abs().quantile(0.50)
            long_thr_data  = new_df.loc[has_rxn, "R_long_5_20"].abs().quantile(0.70)
            # Re-classify with data-driven thresholds
            new_df["reaction_class"] = new_df.apply(
                lambda r: classify_reaction(r["R_short_0_1"], r["R_long_5_20"],
                                            short_thr_data, long_thr_data), axis=1
            )
            log.info(f"  Thresholds recalculated: short={short_thr_data:.4f} long={long_thr_data:.4f}")

        # Merge with any checkpoint rows
        if checkpoint_csv.exists():
            existing = pd.read_csv(checkpoint_csv)
            new_df = pd.concat([existing, new_df], ignore_index=True)

        new_df.drop_duplicates(subset=["accession"], inplace=True)
        new_df.to_csv(output_csv, index=False)

        n_delayed = (new_df["reaction_class"] == "Delayed").sum()
        log.info(f"")
        log.info(f"  Year {year} complete:")
        log.info(f"    Total filings processed : {len(new_df)}")
        log.info(f"    Delayed reactions        : {n_delayed} ({n_delayed/len(new_df):.1%})")
        log.info(f"    LLM errors               : {new_df['llm_error'].notna().sum()}")
        log.info(f"    Saved → {output_csv}")
    else:
        log.warning(f"  No rows produced for {year}")


def merge_all_years(years: list[int]):
    """Merge all year outputs + 2023 master_scored.csv into backfill_master.csv."""
    dfs = []

    # Include 2023 (already processed)
    master_2023 = DATA_DIR / "master_scored.csv"
    if master_2023.exists():
        df_2023 = pd.read_csv(master_2023)
        df_2023["year"] = 2023
        dfs.append(df_2023)
        log.info(f"  Included 2023: {len(df_2023)} rows")
    else:
        log.warning("  master_scored.csv not found — 2023 data not included")

    # Include backfill years
    for year in years:
        out_csv = DATA_DIR / str(year) / "backfill_labelled.csv"
        if out_csv.exists():
            df = pd.read_csv(out_csv)
            dfs.append(df)
            log.info(f"  Included {year}: {len(df)} rows")
        else:
            log.warning(f"  {year}/backfill_labelled.csv not found — skipping")

    if not dfs:
        log.error("No data to merge")
        return

    master = pd.concat(dfs, ignore_index=True)
    master.drop_duplicates(subset=["accession"], inplace=True)
    master_path = DATA_DIR / "backfill_master.csv"
    master.to_csv(master_path, index=False)

    n_total   = len(master)
    n_delayed = (master["reaction_class"] == "Delayed").sum()

    log.info(f"")
    log.info(f"{'='*60}")
    log.info(f"  BACKFILL MASTER SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  Total labelled filings : {n_total}")
    log.info(f"  Delayed (target class) : {n_delayed} ({n_delayed/n_total:.1%})")
    log.info(f"  Years covered          : {sorted(master['year'].unique().tolist())}")
    log.info(f"  Saved → {master_path}")
    log.info(f"")
    log.info(f"  Load for ML training:")
    log.info(f"    df = pd.read_csv('data/backfill_master.csv')")
    log.info(f"    X  = df[feature_cols]")
    log.info(f"    y  = (df['reaction_class'] == 'Delayed').astype(int)")


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill historical 8-K filings for ML training data"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2022, 2021, 2020],
        help="Years to backfill (default: 2022 2021 2020)"
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip Claude API calls (text features only, no cost)"
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip processing, just merge existing year outputs into backfill_master.csv"
    )
    args = parser.parse_args()

    if args.merge_only:
        log.info("Merge-only mode")
        merge_all_years(args.years)
    else:
        for year in sorted(args.years, reverse=True):  # most recent first
            run_backfill_year(year, skip_llm=args.skip_llm)
        merge_all_years(args.years)
