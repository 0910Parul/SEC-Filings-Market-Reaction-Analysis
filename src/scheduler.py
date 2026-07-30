"""
scheduler.py — Automated 8-K ingestion + MOS scoring + alert system
Not Yet Priced In · Team 10

Polls SEC EDGAR every 15 minutes for new 8-K filings from the 50 tracked tickers.
For each new filing:
  1. Downloads and parses the text (parse_filing.py)
  2. Computes A2 structural features
  3. Calls Claude API for importance score (A3)
  4. Scores with MOS_prospective using saved calibration table
  5. Sends Slack/email alert if MOS > threshold
  6. Schedules a 20-day follow-up to pull market reaction and write label back

Usage:
  pip install apscheduler anthropic requests pandas yfinance
  export ANTHROPIC_API_KEY=your_key
  export SLACK_WEBHOOK_URL=your_webhook   # optional
  python scheduler.py
"""

import os, sys, json, time, logging, re, requests
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger

sys.path.insert(0, str(Path(__file__).parent))
from parse_filing import parse_filing, get_primary_item

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent / 'data'
CALIBRATION_CSV= DATA_DIR / 'calibration_table.csv'
SEEN_FILE      = DATA_DIR / 'scheduler_seen.json'
PENDING_FILE   = DATA_DIR / 'scheduler_pending_labels.json'
LOG_CSV        = DATA_DIR / 'scheduler_log.csv'

MOS_ALERT_THRESHOLD = 0.65   # send alert if MOS_prospective >= this
POLL_INTERVAL_MIN   = 15     # how often to check EDGAR
LABEL_DELAY_DAYS    = 22     # trading days to wait before pulling reaction label

HEADERS = {'User-Agent': 'Team10 MSBA contact@umn.edu', 'Accept-Encoding': 'gzip, deflate'}

TICKERS_TO_CIK = {
    'AAPL':'0000320193','MSFT':'0000789019','GOOGL':'0001652044','AMZN':'0001018724',
    'META':'0001326801','NVDA':'0001045810','ADBE':'0000796343','CRM':'0001108524',
    'JPM':'0000019617','BAC':'0000070858','GS':'0000886982','MS':'0000895421',
    'WFC':'0000072971','BLK':'0001364742','JNJ':'0000200406','PFE':'0000078003',
    'MRK':'0000310158','ABBV':'0001551152','UNH':'0000731766','LLY':'0000059478',
    'WMT':'0000104169','COST':'0000909832','KO':'0000021344','PEP':'0000077476',
    'NKE':'0000320187','SBUX':'0000829224','BA':'0000012927','GE':'0000040987',
    'HON':'0000773840','CAT':'0000018230','UNP':'0000100885','XOM':'0000034088',
    'CVX':'0000093410','COP':'0001163165','SLB':'0000087347','DIS':'0001001039',
    'NFLX':'0001065280','CMCSA':'0001166691','T':'0000732717','VZ':'0000732712',
    'MMM':'0000066740','DOW':'0001751788','DD':'0001666700','FCX':'0000831259',
    'TSLA':'0001318605','INTC':'0000050863','ORCL':'0001341439','QCOM':'0000804328',
    'AXP':'0000004962','CVS':'0000064803',
}

# ── Seen-filing tracker ─────────────────────────────────────────────────────
def load_seen():
    if SEEN_FILE.exists():
        return set(json.loads(SEEN_FILE.read_text()))
    return set()

def save_seen(seen):
    SEEN_FILE.write_text(json.dumps(list(seen)))

def load_pending():
    if PENDING_FILE.exists():
        return json.loads(PENDING_FILE.read_text())
    return []

def save_pending(pending):
    PENDING_FILE.write_text(json.dumps(pending))

# ── EDGAR polling ───────────────────────────────────────────────────────────
def get_recent_8k(ticker, cik):
    url = f'https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json'
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f'EDGAR fetch failed for {ticker}: {e}')
        return []

    recent = data.get('filings', {}).get('recent', {})
    forms        = recent.get('form', [])
    dates        = recent.get('filingDate', [])
    accessions   = recent.get('accessionNumber', [])
    primary_docs = recent.get('primaryDocument', [])
    company      = data.get('name', ticker)

    cutoff = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    results = []
    for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
        if form != '8-K' or date < cutoff:
            continue
        acc_clean = acc.replace('-', '')
        cik_int   = int(cik)
        doc_url   = f'https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{doc}'
        results.append({
            'ticker': ticker, 'cik': cik, 'company': company,
            'accession': acc, 'filed_date': date, 'doc_url': doc_url,
        })
    return results

# ── Text features (A2) ──────────────────────────────────────────────────────
FORWARD_TERMS = {
    'will','expect','expects','expected','anticipate','anticipates',
    'forecast','forecasts','project','projects','projected',
    'estimate','estimates','estimated','intend','intends',
    'plan','plans','planned','believe','believes',
    'outlook','guidance','target','targets','goal','goals',
    'upcoming','future','forward','should','could','may','might','would',
    'pending','approximately',
}

def compute_text_features(clean_text):
    if not clean_text or not isinstance(clean_text, str):
        return {'numeric_density': 0, 'forward_looking_density': 0,
                'financial_symbol_density': 0, 'baseline_importance': 0.5}
    words = clean_text.split()
    if not words:
        return {'numeric_density': 0, 'forward_looking_density': 0,
                'financial_symbol_density': 0, 'baseline_importance': 0.5}
    n = len(words)
    nd  = sum(1 for w in words if any(c.isdigit() for c in w)) / n * 100
    fld = sum(1 for w in words if w.strip('.,;:!?"\'()[]').lower() in FORWARD_TERMS) / n * 100
    fsd = (clean_text.count('$') + clean_text.count('%')) / n * 100
    bi  = min(1.0, (nd/10*0.33 + fld/8*0.33 + fsd/5*0.34))
    return {'numeric_density': nd, 'forward_looking_density': fld,
            'financial_symbol_density': fsd, 'baseline_importance': bi}

# ── LLM scoring (A3) ────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    'You are a financial analyst reading SEC 8-K filings. '
    'Return valid JSON with: event_category (one of Earnings/Material Agreement/'
    'Executive Change/Voting Results/Governance/Regulatory/Financial Obligation/'
    'Acquisition/Disposition/Securities/Accounting/Other), importance_score (1-5 int), '
    'key_signals (2-5 VERBATIM quotes from text, each under 30 words), '
    'reasoning (2-3 sentences), confidence (0.0-1.0). '
    'Return only the JSON object, no markdown.'
)

def llm_score(clean_text, max_chars=15000):
    import anthropic
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model='claude-sonnet-4-5',
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{'role':'user','content':
                f'Analyze this 8-K filing:\n\n{clean_text[:max_chars]}'}],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r'^```[a-z]*\n?','',raw); raw = re.sub(r'```$','',raw).strip()
        r = json.loads(raw)
        return {
            'llm_event_category': r.get('event_category','Other'),
            'importance_score':   r.get('importance_score', 3),
            'key_signals':        r.get('key_signals', []),
            'reasoning':          r.get('reasoning', ''),
            'llm_confidence':     r.get('confidence', 0.5),
            'error': None,
        }
    except Exception as e:
        log.warning(f'LLM scoring failed: {e}')
        return {'llm_event_category':'Other','importance_score':3,
                'key_signals':[],'reasoning':'','llm_confidence':0.0,'error':str(e)}

# ── MOS_prospective ─────────────────────────────────────────────────────────
def build_lift_lookup(calib_df):
    lookup = {}
    for _, row in calib_df.iterrows():
        feat = row['feature']; val = str(row['value'])
        lift = row['lift'] if pd.notna(row.get('lift')) else 1.0
        lookup.setdefault(feat, {})[val] = lift
    return lookup

def get_tier(value, lo, hi):
    if pd.isna(value): return 'mid'
    return 'low' if value <= lo else 'high' if value >= hi else 'mid'

def mos_prospective(features, lift_lookup, tier_lo_hi):
    prior = 1.0
    cat = str(features.get('broad_category','Other'))
    prior *= lift_lookup.get('broad_category',{}).get(cat, 1.0)

    imp_norm = (features.get('importance_score',3) - 1) / 4
    tier = get_tier(imp_norm, *tier_lo_hi['llm_importance_tier'])
    prior *= lift_lookup.get('llm_importance_tier',{}).get(tier, 1.0)

    tier = get_tier(features.get('forward_looking_density',0), *tier_lo_hi['forward_looking_tier'])
    prior *= lift_lookup.get('forward_looking_tier',{}).get(tier, 1.0)

    tier = get_tier(features.get('numeric_density',0), *tier_lo_hi['numeric_density_tier'])
    prior *= lift_lookup.get('numeric_density_tier',{}).get(tier, 1.0)

    tier = get_tier(features.get('baseline_importance',0.5), *tier_lo_hi['baseline_importance_tier'])
    prior *= lift_lookup.get('baseline_importance_tier',{}).get(tier, 1.0)

    return prior

# ── Alert ───────────────────────────────────────────────────────────────────
def send_alert(filing, mos, key_signals, reasoning):
    webhook = os.environ.get('SLACK_WEBHOOK_URL')
    msg = (
        f":bell: *Not Yet Priced In — High MOS Alert*\n"
        f"*{filing['ticker']}* · {filing['filed_date']} · {filing.get('broad_category','')}\n"
        f"MOS_prospective: *{mos:.3f}* (top {(1-mos)*100:.0f}% of historical Delayed filings)\n"
        f"LLM importance: {filing.get('importance_score','?')}/5\n\n"
        f"*Key signals:*\n" +
        '\n'.join(f"  - {s[:100]}" for s in (key_signals or [])[:3]) +
        f"\n\n*Reasoning:* {reasoning[:200]}\n"
        f"Filing URL: {filing.get('doc_url','')}"
    )
    log.info(f"ALERT: {filing['ticker']} MOS={mos:.3f}")
    log.info(msg)
    if webhook:
        try:
            requests.post(webhook, json={'text': msg}, timeout=10)
            log.info('Slack alert sent')
        except Exception as e:
            log.warning(f'Slack send failed: {e}')

# ── Market reaction follow-up ───────────────────────────────────────────────
def fetch_reaction_label(ticker, day0_str):
    try:
        import yfinance as yf
        d0 = pd.Timestamp(day0_str)
        fetch_end = (d0 + timedelta(days=35)).strftime('%Y-%m-%d')
        prices = yf.download(ticker, start=(d0 - timedelta(days=3)).strftime('%Y-%m-%d'),
                              end=fetch_end, auto_adjust=True, progress=False)['Close'].squeeze()
        bench  = yf.download('^GSPC', start=(d0 - timedelta(days=3)).strftime('%Y-%m-%d'),
                              end=fetch_end, auto_adjust=True, progress=False)['Close'].squeeze()
        if prices.empty or bench.empty:
            return None
        ar = (prices.pct_change() - bench.pct_change().reindex(prices.index).ffill())
        trading_days = ar.dropna().index
        idx = trading_days.searchsorted(d0)
        if idx + 20 >= len(trading_days):
            return None
        r_short = float(ar.iloc[idx:idx+2].sum())
        r_long  = float(ar.iloc[idx+5:idx+21].sum())
        same_dir = (r_short * r_long) > 0
        short_thr, long_thr = 0.01, 0.03
        if abs(r_short) >= short_thr:
            label = 'Immediate'
        elif abs(r_short) < short_thr and abs(r_long) > long_thr and same_dir:
            label = 'Delayed'
        elif same_dir:
            label = 'Gradual'
        else:
            label = 'No Reaction'
        return {'R_short': round(r_short,5), 'R_long': round(r_long,5), 'reaction_class': label}
    except Exception as e:
        log.warning(f'Reaction fetch failed {ticker}: {e}')
        return None

# ── Main poll job ───────────────────────────────────────────────────────────
def load_scoring_assets():
    if not CALIBRATION_CSV.exists():
        raise FileNotFoundError(f'calibration_table.csv not found at {CALIBRATION_CSV}')
    calib = pd.read_csv(CALIBRATION_CSV)
    lift_lookup = build_lift_lookup(calib)

    # Fixed tier thresholds (approximate corpus percentiles)
    tier_lo_hi = {
        'forward_looking_tier':     (1.2, 3.1),
        'numeric_density_tier':     (2.8, 7.4),
        'baseline_importance_tier': (0.38, 0.62),
        'llm_importance_tier':      (0.33, 0.67),
    }
    corpus_prior_mean = 1.0
    return lift_lookup, tier_lo_hi, corpus_prior_mean

def poll_and_score():
    log.info('Polling EDGAR for new 8-K filings...')
    try:
        lift_lookup, tier_lo_hi, _ = load_scoring_assets()
    except Exception as e:
        log.error(f'Could not load scoring assets: {e}'); return

    seen = load_seen()
    pending = load_pending()
    new_rows = []

    for ticker, cik in TICKERS_TO_CIK.items():
        filings = get_recent_8k(ticker, cik)
        time.sleep(0.15)
        for f in filings:
            uid = f['accession']
            if uid in seen:
                continue
            seen.add(uid)
            log.info(f"New filing: {ticker} {f['filed_date']} {uid}")

            # Download and parse
            try:
                resp = requests.get(f['doc_url'], headers=HEADERS, timeout=20)
                resp.raise_for_status()
                from bs4 import BeautifulSoup
                text_raw = BeautifulSoup(resp.text, 'lxml').get_text(' ', strip=True)[:20000]
            except Exception as e:
                log.warning(f'Download failed {ticker}: {e}'); continue

            parsed = parse_filing(text_raw)
            primary = get_primary_item(parsed)
            clean   = parsed.get('clean_text','')

            text_feats = compute_text_features(clean)
            llm        = llm_score(clean)

            broad_map = {
                '2.02':'Earnings','1.01':'Material Agreement','1.02':'Material Agreement',
                '5.02':'Executive Change','5.07':'Voting Results','7.01':'Regulatory',
                '8.01':'Regulatory','2.01':'Acquisition/Disposition','5.03':'Governance',
            }
            broad_cat = broad_map.get(primary['number'] if primary else '', 'Other')

            features = {
                'broad_category':          broad_cat,
                'importance_score':        llm['importance_score'],
                'forward_looking_density': text_feats['forward_looking_density'],
                'numeric_density':         text_feats['numeric_density'],
                'baseline_importance':     text_feats['baseline_importance'],
            }

            prior = mos_prospective(features, lift_lookup, tier_lo_hi)
            # Normalise to [0,1] using sigmoid-like mapping (no corpus needed)
            mos_p = float(1 / (1 + np.exp(-2 * (prior - 1.0))))

            log.info(f'  Scored: MOS_prospective={mos_p:.3f}  importance={llm["importance_score"]}')

            row = {
                'ticker': ticker, 'filed_date': f['filed_date'],
                'accession': uid, 'doc_url': f['doc_url'],
                'company': f['company'], 'broad_category': broad_cat,
                'importance_score': llm['importance_score'],
                'llm_event_category': llm['llm_event_category'],
                'key_signals': json.dumps(llm['key_signals']),
                'reasoning': llm['reasoning'],
                'llm_confidence': llm['llm_confidence'],
                **text_feats,
                'MOS_prospective': mos_p,
                'scored_at': datetime.now().isoformat(),
                'reaction_class': None, 'R_short': None, 'R_long': None,
            }
            new_rows.append(row)

            if mos_p >= MOS_ALERT_THRESHOLD:
                send_alert(
                    {**f, 'broad_category': broad_cat, 'importance_score': llm['importance_score']},
                    mos_p, llm['key_signals'], llm['reasoning']
                )
                # Schedule label pull in LABEL_DELAY_DAYS days
                label_time = datetime.now() + timedelta(days=LABEL_DELAY_DAYS)
                pending.append({'ticker': ticker, 'filed_date': f['filed_date'],
                                'accession': uid, 'label_at': label_time.isoformat()})
                log.info(f'  Label follow-up scheduled for {label_time.date()}')

    save_seen(seen)
    save_pending(pending)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if LOG_CSV.exists():
            existing = pd.read_csv(LOG_CSV)
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_csv(LOG_CSV, index=False)
        log.info(f'Logged {len(new_rows)} new filings to {LOG_CSV}')
    else:
        log.info('No new filings found this poll.')

# ── Label follow-up job ─────────────────────────────────────────────────────
def check_pending_labels():
    pending = load_pending()
    if not pending:
        return
    now = datetime.now()
    still_pending = []
    for item in pending:
        if pd.Timestamp(item['label_at']) <= now:
            log.info(f"Fetching label for {item['ticker']} {item['filed_date']}")
            result = fetch_reaction_label(item['ticker'], item['filed_date'])
            if result:
                log.info(f"  Label: {result['reaction_class']}  R_long={result['R_long']*100:.2f}%")
                if LOG_CSV.exists():
                    df = pd.read_csv(LOG_CSV)
                    mask = df['accession'] == item['accession']
                    df.loc[mask, 'reaction_class'] = result['reaction_class']
                    df.loc[mask, 'R_short']        = result['R_short']
                    df.loc[mask, 'R_long']          = result['R_long']
                    df.to_csv(LOG_CSV, index=False)
                if result['reaction_class'] == 'Delayed':
                    log.info(f"  *** CONFIRMED DELAYED REACTION: {item['ticker']} {item['filed_date']} ***")
            else:
                still_pending.append(item)
        else:
            still_pending.append(item)
    save_pending(still_pending)

# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    DATA_DIR.mkdir(exist_ok=True)
    log.info('Not Yet Priced In — Scheduler starting')
    log.info(f'Tracking {len(TICKERS_TO_CIK)} tickers')
    log.info(f'Poll interval: every {POLL_INTERVAL_MIN} minutes')
    log.info(f'Alert threshold: MOS_prospective >= {MOS_ALERT_THRESHOLD}')
    log.info(f'Calibration file: {CALIBRATION_CSV}')

    scheduler = BlockingScheduler(timezone='America/New_York')

    scheduler.add_job(
        poll_and_score,
        trigger=IntervalTrigger(minutes=POLL_INTERVAL_MIN),
        id='poll_edgar',
        name='Poll EDGAR for new 8-Ks',
        next_run_time=datetime.now(),  # run immediately on start
        misfire_grace_time=300,
    )

    scheduler.add_job(
        check_pending_labels,
        trigger=IntervalTrigger(hours=6),
        id='check_labels',
        name='Check pending reaction labels',
        next_run_time=datetime.now() + timedelta(seconds=30),
        misfire_grace_time=3600,
    )

    log.info('Scheduler started. Press Ctrl+C to stop.')
    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info('Scheduler stopped.')
