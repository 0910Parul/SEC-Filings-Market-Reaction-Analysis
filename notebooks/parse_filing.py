"""
Stage A0: 8-K Document Structuring
──────────────────────────────────
Parse raw 8-K filing text into a structured dict:
- items: list of {number, title, content}
- metadata: company, filing_date, ticker (if derivable)
- noise_removed: list of detected boilerplate sections

Key design decisions:
- Use Item X.XX headers as structural anchors (SEC-standardized)
- Drop Item 9.01 (Exhibits) automatically — no informational content
- Detect and drop Cautionary Statements / Non-GAAP disclosures
- Drop SEC header boilerplate up to the first real Item
- Drop SIGNATURES block and everything after
"""

import re
from pathlib import Path


# ─── SEC Item Number → Event Category Mapping (Rule-based) ─────────────
# Source: SEC Form 8-K official instructions
ITEM_CATEGORIES = {
    # Section 1: Registrant's Business and Operations
    "1.01": "Material Agreement (Entry)",
    "1.02": "Material Agreement (Termination)",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety",
    # Section 2: Financial Information
    "2.01": "Acquisition/Disposition of Assets",
    "2.02": "Results of Operations / Earnings",
    "2.03": "Direct Financial Obligation",
    "2.04": "Triggering Event (Off-BS / Obligation)",
    "2.05": "Costs from Exit/Disposal",
    "2.06": "Material Impairments",
    # Section 3: Securities and Trading Markets
    "3.01": "Notice of Delisting / Listing Standards",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    # Section 4: Matters Related to Accountants and Financial Statements
    "4.01": "Change in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previous Financial Statements",
    # Section 5: Corporate Governance and Management
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure/Election of Directors or Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.04": "Temporary Suspension of Trading",
    "5.05": "Amendment to Code of Ethics",
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    # Section 6: Asset-Backed Securities (rare for large caps)
    # Section 7: Regulation FD
    "7.01": "Regulation FD Disclosure",
    # Section 8: Other Events
    "8.01": "Other Events",
    # Section 9: Financial Statements and Exhibits
    "9.01": "Financial Statements and Exhibits",  # Almost always empty — drop
}

# Items that are structural/administrative and carry no substantive content
STRUCTURAL_ONLY_ITEMS = {"9.01"}


# ─── Core Parser ───────────────────────────────────────────────────────

# Item header regex:
# Key insight from inspecting real filings:
#   - REAL headers:     "Item 2.02.  Results of Operations..."  (followed by UPPERCASE title)
#   - FALSE references: "in this Item 2.02 by reference"        (followed by lowercase)
#
# Note: we DON'T use re.IGNORECASE globally because it would make [A-Z] match
# lowercase too, breaking our lookahead. Instead we use inline (?i:...) only
# for the "Item" word, keeping [A-Z] case-sensitive.
ITEM_HEADER_RE = re.compile(
    r"""
    \b(?i:Item)               # "Item" case-insensitive (but only this part)
    \s+
    (\d+\.\d+)                # capture: e.g. 2.02
    [\s\.]*?                  # optional punctuation/whitespace (non-greedy)
    (?=[A-Z][a-z])            # lookahead: next non-space is TRULY uppercase
                              #   followed by lowercase. "by reference" fails.
                              #   "Results of Operations" passes.
    """,
    re.VERBOSE,
)

# Boilerplate section markers (case-insensitive)
# Content AFTER any of these markers is considered noise
BOILERPLATE_END_MARKERS = [
    r"Cautionary\s+Statement\s+Concerning\s+Forward-Looking\s+Statements",
    r"Forward-Looking\s+Statements?\s*$",  # standalone header
    r"Safe\s+Harbor\s+Statement",
    r"Non-GAAP\s+Financial\s+Measures?",  # typically followed by long disclaimer
    r"SIGNATURES?\s*$",  # end of document
]

BOILERPLATE_END_RE = re.compile(
    r"(?:" + r"|".join(BOILERPLATE_END_MARKERS) + r")",
    re.IGNORECASE | re.MULTILINE,
)

# SEC header boilerplate markers — content BEFORE the first Item is always boilerplate
# but we use this to validate we've found a real filing
SEC_HEADER_MARKERS = [
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
    "FORM 8-K",
    "CURRENT REPORT",
]


def parse_filing(raw_text: str) -> dict:
    """
    Parse a raw 8-K filing text into structured components.

    Returns
    -------
    dict with keys:
        items:           list of {number, category, title, content, is_structural}
        substantive_items: list of items excluding structural-only (e.g. 9.01)
        clean_text:      concatenated content of substantive items (for downstream analysis)
        noise_removed:   list of str describing what was dropped
        raw_length:      char count of input
        clean_length:    char count of clean_text
        reduction_ratio: 1 - (clean_length / raw_length)
    """
    noise_removed = []
    raw_length = len(raw_text)

    # Step 1: Drop content after first boilerplate end marker
    boilerplate_match = BOILERPLATE_END_RE.search(raw_text)
    if boilerplate_match:
        body_text = raw_text[: boilerplate_match.start()]
        dropped_section = raw_text[boilerplate_match.start():]
        noise_removed.append(
            f"Dropped {len(dropped_section)} chars starting at "
            f"'{boilerplate_match.group()[:60]}...'"
        )
    else:
        body_text = raw_text

    # Step 2: Find all Item headers and their positions
    # We collect (position, item_number) tuples
    item_matches = list(ITEM_HEADER_RE.finditer(body_text))

    if not item_matches:
        return {
            "items": [],
            "substantive_items": [],
            "clean_text": "",
            "noise_removed": noise_removed + ["No Item headers found"],
            "raw_length": raw_length,
            "clean_length": 0,
            "reduction_ratio": 1.0,
            "parse_success": False,
            "parse_confidence": 0.0,
            "confidence_reason": "no_item_headers_found",
        }

    # Step 3: Extract content between consecutive Item headers
    items = []
    for i, match in enumerate(item_matches):
        item_num = match.group(1)
        start = match.end()
        end = item_matches[i + 1].start() if i + 1 < len(item_matches) else len(body_text)
        content = body_text[start:end].strip()

        category = ITEM_CATEGORIES.get(item_num, "Unknown")
        is_structural = item_num in STRUCTURAL_ONLY_ITEMS

        # Try to extract a title — typically the first sentence after the Item header
        # Item headers commonly look like: "Item 2.02. Results of Operations and Financial Condition."
        title_match = re.match(r"\s*\.?\s*([^\.]+?)\.", content)
        title = title_match.group(1).strip() if title_match else ""

        # Remove the title portion from content (it's duplicated in our category field)
        if title_match:
            content_body = content[title_match.end():].strip()
        else:
            content_body = content

        items.append({
            "number": item_num,
            "category": category,
            "title": title,
            "content": content_body,
            "is_structural": is_structural,
            "content_length": len(content_body),
        })

    # Step 4: Filter to substantive items
    substantive_items = [it for it in items if not it["is_structural"]]

    # Step 5: Concatenate substantive content into clean_text
    clean_text = "\n\n".join(
        f"[Item {it['number']} — {it['category']}]\n{it['content']}"
        for it in substantive_items
    )

    # Step 6: Note what's structurally skipped
    for it in items:
        if it["is_structural"]:
            noise_removed.append(
                f"Skipped Item {it['number']} ({it['category']}) — "
                f"{it['content_length']} chars of exhibit listing"
            )

    # Step 7: Note pre-header boilerplate (SEC legal header)
    first_item_pos = item_matches[0].start()
    if first_item_pos > 200:
        noise_removed.append(
            f"Dropped {first_item_pos} chars of SEC header boilerplate "
            f"(before first Item)"
        )

    clean_length = len(clean_text)
    reduction_ratio = round(1 - clean_length / raw_length, 3) if raw_length else 0

    # Compute parse confidence — a heuristic 0-1 score indicating "does this
    # parse output look plausible?". Not an absolute correctness measure; it's
    # a flag for which filings deserve manual review.
    confidence, confidence_reason = _compute_parse_confidence(
        parse_success=True,
        raw_length=raw_length,
        clean_length=clean_length,
        reduction_ratio=reduction_ratio,
        substantive_items=substantive_items,
        all_items=items,
    )

    return {
        "items": items,
        "substantive_items": substantive_items,
        "clean_text": clean_text,
        "noise_removed": noise_removed,
        "raw_length": raw_length,
        "clean_length": clean_length,
        "reduction_ratio": reduction_ratio,
        "parse_success": True,
        "parse_confidence": confidence,
        "confidence_reason": confidence_reason,
    }


def _compute_parse_confidence(
    parse_success: bool,
    raw_length: int,
    clean_length: int,
    reduction_ratio: float,
    substantive_items: list,
    all_items: list,
) -> tuple[float, str]:
    """
    Heuristic confidence score for a parse result.
    Returns (score 0-1, reason string).
    Lower score = more likely to need manual review.
    """
    if not parse_success:
        return 0.0, "parse_failed"
    if len(all_items) == 0:
        return 0.0, "no_items_found"
    if len(substantive_items) == 0:
        return 0.2, "only_structural_items (e.g., 9.01 only)"
    if clean_length < 100:
        return 0.3, f"clean_text_too_short ({clean_length} chars)"
    if reduction_ratio > 0.95 and raw_length > 2000:
        return 0.4, f"over_cleaned (reduction {reduction_ratio:.1%})"
    if reduction_ratio < 0.30 and raw_length > 3000:
        return 0.5, f"under_cleaned (reduction {reduction_ratio:.1%})"
    if len(all_items) > 5:
        return 0.6, f"too_many_items ({len(all_items)}) — possible false matches"
    # Check for unknown item categories
    unknown = [it["number"] for it in all_items if it["category"] == "Unknown"]
    if unknown:
        return 0.7, f"unknown_item_codes: {unknown}"
    return 1.0, "ok"


def get_primary_item(parsed: dict) -> dict | None:
    """
    For filings with multiple substantive items, pick the 'primary' one
    based on priority: market-moving items > administrative items.

    Priority order (higher index = higher priority):
    - 8.01 (Other) < 7.01 (Reg FD) < 5.x (Governance) < 2.x (Financial) < 1.x (Material)
    """
    if not parsed["substantive_items"]:
        return None
    if len(parsed["substantive_items"]) == 1:
        return parsed["substantive_items"][0]

    # Priority: lower section number = higher priority (1.x beats 8.x)
    # Within substantive, prefer longer content (more information)
    def priority_key(item):
        section = int(item["number"].split(".")[0])
        # Invert so lower section = higher priority
        return (-section, item["content_length"])

    return max(parsed["substantive_items"], key=priority_key)


# ─── Test harness ──────────────────────────────────────────────────────

def batch_parse(
    filings_dir: str,
    output_csv: str | None = None,
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    Parse all .txt files in filings_dir and return a DataFrame with one row
    per filing plus aggregate diagnostics.

    If output_csv is given, writes the summary DataFrame there.
    Also writes {output_csv}_clean_text.csv with ticker, accession, clean_text
    for downstream Stage A2/A3 consumption.
    """
    import pandas as pd

    filings_dir = Path(filings_dir)
    files = sorted(filings_dir.glob("*.txt"))
    if verbose:
        print(f"Parsing {len(files)} filings from {filings_dir}...")

    records = []
    clean_records = []
    for i, f in enumerate(files):
        try:
            raw = f.read_text(encoding="utf-8")
        except Exception as e:
            records.append({
                "filename": f.name,
                "parse_success": False,
                "parse_confidence": 0.0,
                "confidence_reason": f"read_error: {e}",
                "raw_length": 0,
                "clean_length": 0,
                "reduction_ratio": 0.0,
                "n_items": 0,
                "n_substantive": 0,
                "primary_item": None,
                "primary_category": None,
                "all_items": "",
            })
            continue

        parsed = parse_filing(raw)
        primary = get_primary_item(parsed)

        # Extract ticker and accession from filename (format: TICKER_ACCESSION.txt)
        parts = f.stem.split("_", 1)
        ticker = parts[0] if parts else ""
        accession = parts[1] if len(parts) > 1 else ""

        records.append({
            "filename": f.name,
            "ticker": ticker,
            "accession": accession,
            "parse_success": parsed["parse_success"],
            "parse_confidence": parsed["parse_confidence"],
            "confidence_reason": parsed["confidence_reason"],
            "raw_length": parsed["raw_length"],
            "clean_length": parsed["clean_length"],
            "reduction_ratio": parsed["reduction_ratio"],
            "n_items": len(parsed["items"]),
            "n_substantive": len(parsed["substantive_items"]),
            "primary_item": primary["number"] if primary else None,
            "primary_category": primary["category"] if primary else None,
            "all_items": ",".join(it["number"] for it in parsed["items"]),
        })

        clean_records.append({
            "filename": f.name,
            "ticker": ticker,
            "accession": accession,
            "primary_item": primary["number"] if primary else None,
            "primary_category": primary["category"] if primary else None,
            "clean_text": parsed["clean_text"],
        })

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(files)}")

    df = pd.DataFrame(records)
    clean_df = pd.DataFrame(clean_records)

    if output_csv:
        df.to_csv(output_csv, index=False)
        base = output_csv.rsplit(".", 1)[0]
        clean_df.to_csv(f"{base}_clean_text.csv", index=False)
        if verbose:
            print(f"\nSaved diagnostic summary → {output_csv}")
            print(f"Saved clean text       → {base}_clean_text.csv")

    if verbose:
        print_diagnostic_report(df)

    return df


def print_diagnostic_report(df: "pd.DataFrame") -> None:
    """Print a structured parse-quality report."""
    import pandas as pd

    n = len(df)
    n_success = df["parse_success"].sum()
    n_fail = n - n_success

    print("\n" + "═" * 70)
    print("  PARSE QUALITY REPORT")
    print("═" * 70)
    print(f"  Total filings:       {n}")
    print(f"  Parse success:       {n_success} ({n_success/n:.1%})")
    print(f"  Parse failure:       {n_fail} ({n_fail/n:.1%})")

    # Confidence distribution
    print(f"\n  ── Parse confidence distribution ──")
    bins = [0, 0.3, 0.5, 0.7, 0.99, 1.01]
    labels = ["0.0-0.3 (critical)", "0.3-0.5 (poor)",
              "0.5-0.7 (suspect)", "0.7-0.99 (ok-ish)", "1.0 (clean)"]
    df["conf_bucket"] = pd.cut(df["parse_confidence"], bins=bins,
                                labels=labels, include_lowest=True)
    for bucket, count in df["conf_bucket"].value_counts().sort_index().items():
        print(f"    {bucket:25s} {count:>4} filings ({count/n:.1%})")

    # Reason breakdown for non-perfect confidence
    print(f"\n  ── Confidence reasons (non-OK only) ──")
    issues = df[df["parse_confidence"] < 1.0]["confidence_reason"].value_counts()
    for reason, count in issues.items():
        print(f"    {count:>4}  {reason}")

    # Reduction ratio distribution
    print(f"\n  ── Reduction ratio distribution ──")
    ratio_bins = [0, 0.3, 0.7, 0.9, 0.95, 1.01]
    ratio_labels = ["<30% (under-cleaned)", "30-70%", "70-90% (normal)",
                    "90-95%", ">95% (over-cleaned)"]
    df["ratio_bucket"] = pd.cut(df["reduction_ratio"], bins=ratio_bins,
                                 labels=ratio_labels, include_lowest=True)
    for bucket, count in df["ratio_bucket"].value_counts().sort_index().items():
        print(f"    {bucket:25s} {count:>4} filings")

    # Clean length distribution
    print(f"\n  ── Clean length distribution ──")
    length_bins = [0, 100, 500, 2000, 10000, float("inf")]
    length_labels = ["<100 (too short)", "100-500", "500-2000 (normal)",
                     "2000-10000 (rich)", ">10000"]
    df["length_bucket"] = pd.cut(df["clean_length"], bins=length_bins,
                                  labels=length_labels, include_lowest=True)
    for bucket, count in df["length_bucket"].value_counts().sort_index().items():
        print(f"    {bucket:25s} {count:>4} filings")

    # Primary item category breakdown
    print(f"\n  ── Primary item categories ──")
    for cat, count in df["primary_category"].value_counts().head(15).items():
        print(f"    {count:>4}  {cat}")

    # Unknown item codes
    all_items_flat = []
    for items_str in df["all_items"].dropna():
        all_items_flat.extend(items_str.split(","))
    unknown_codes = [c for c in all_items_flat
                     if c and c not in ITEM_CATEGORIES]
    if unknown_codes:
        from collections import Counter
        print(f"\n  ── Unknown item codes (need category mapping) ──")
        for code, count in Counter(unknown_codes).most_common():
            print(f"    {count:>4}  Item {code}")

    # Stratified sampling for manual review
    print(f"\n  ── Suggested manual review (stratified sample) ──")
    samples_to_show = [
        ("parse failures", df[~df["parse_success"]]),
        ("over-cleaned (>95%)", df[df["reduction_ratio"] > 0.95]),
        ("under-cleaned (<30%, raw>3k)",
         df[(df["reduction_ratio"] < 0.30) & (df["raw_length"] > 3000)]),
        ("too-short clean (<100 chars)",
         df[(df["clean_length"] < 100) & df["parse_success"]]),
        ("too-many items (>5)", df[df["n_items"] > 5]),
    ]
    for label, subset in samples_to_show:
        if len(subset) == 0:
            continue
        sample = subset.sample(min(5, len(subset)), random_state=42)
        print(f"\n    ★ {label} ({len(subset)} total, showing up to 5):")
        for _, row in sample.iterrows():
            print(f"       {row['filename']:45s}  "
                  f"raw={row['raw_length']:>6,}  "
                  f"clean={row['clean_length']:>5,}  "
                  f"ratio={row['reduction_ratio']:>5.1%}  "
                  f"items={row['all_items']}")

    print("\n" + "═" * 70)
    print("  DONE — review flagged filings above, then iterate parser if needed")
    print("═" * 70)


def demo_on_file(filepath: str) -> None:
    """Pretty-print parse result for a single file (manual inspection)."""
    raw = Path(filepath).read_text(encoding="utf-8")
    parsed = parse_filing(raw)

    ticker = Path(filepath).name.split("_")[0]
    print(f"\n{'═' * 70}")
    print(f"  {ticker} — {Path(filepath).name}")
    print(f"{'═' * 70}")
    print(f"  Raw length:       {parsed['raw_length']:>6,} chars")
    print(f"  Clean length:     {parsed['clean_length']:>6,} chars")
    print(f"  Reduction:        {parsed['reduction_ratio']:>6.1%}")
    print(f"  Items found:      {len(parsed['items'])}")
    print(f"  Substantive:      {len(parsed['substantive_items'])}")

    print(f"\n  ── Items ──")
    for it in parsed["items"]:
        marker = "✗" if it["is_structural"] else "✓"
        print(f"    {marker} Item {it['number']:<5} [{it['content_length']:>5} chars] "
              f"{it['category']}")

    print(f"\n  ── Noise removed ──")
    for note in parsed["noise_removed"]:
        print(f"    • {note}")

    primary = get_primary_item(parsed)
    if primary:
        print(f"\n  ── Primary item: {primary['number']} ({primary['category']}) ──")
        preview = primary["content"][:300].replace("\n", " ")
        if len(primary["content"]) > 300:
            preview += "..."
        print(f"    {preview}")


if __name__ == "__main__":
    import sys

    sample_files = [
        "/mnt/user-data/uploads/QCOM_0000804328_23_000054.txt",
        "/mnt/user-data/uploads/CVS_0000064803_23_000008.txt",
        "/mnt/user-data/uploads/CVS_0000064803_23_000003.txt",
        "/mnt/user-data/uploads/DD_0001666700_23_000054.txt",
        "/mnt/user-data/uploads/TSLA_0001564590_23_007379.txt",
        "/mnt/user-data/uploads/INTC_0000050863_23_000027.txt",
    ]

    for f in sample_files:
        demo_on_file(f)
