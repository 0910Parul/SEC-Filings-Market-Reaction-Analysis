import os
import sys
import pandas as pd
import numpy as np
import re
from pathlib import Path
from openai import OpenAI
from src import config

# Setup logging
logger = config.setup_logging(__name__)

# Add notebooks directory to sys.path to import parse_filing
sys.path.insert(0, str(config.NOTEBOOKS_DIR))
from parse_filing import parse_filing

SYSTEM_PROMPT = (
    "You are a financial analyst assistant reading SEC 8-K filings.\n\n"
    "Return valid JSON with exactly these fields:\n"
    "- event_category: one of [Earnings, Material Agreement, Executive Change, "
    "Voting Results, Governance, Regulatory, Financial Obligation, "
    "Acquisition/Disposition, Securities, Accounting, Other]\n"
    "- importance_score: integer 1-5\n"
    "- key_signals: list of 2-5 VERBATIM quotes from the filing. "
    "Each must appear EXACTLY in the source text, character-for-character. "
    "Each quote will be programmatically verified and rejected if it does not match verbatim. "
    "Keep each quote under 30 words.\n"
    "- reasoning: 2-3 sentences citing specific text\n"
    "- confidence: float 0.0-1.0\n\n"
    "RUBRIC:\n"
    "5=Very High: specific financials, guidance changes, major M&A or leadership news\n"
    "4=High: concrete updates with numbers or named parties\n"
    "3=Medium: substantive but routine news\n"
    "2=Low: administrative or procedural items\n"
    "1=Very Low: pure formalities, no business impact\n\n"
    "If no verbatim quotes found, return empty key_signals and confidence below 0.5."
)

USER_PROMPT = (
    "Analyze this 8-K filing. Return only the JSON object, no markdown fences.\n\n"
    "---FILING TEXT---\n"
    "{clean_text}\n"
    "---END FILING TEXT---"
)

def compute_structural_features(text):
    """
    Computes structural features from raw text as defined in NB2.
    """
    try:
        logger.info(f"Computing structural features for text (len: {len(text)})")
        if not text or not isinstance(text, str):
            return 0.0, 0.0, 0.0
        
        words = text.split()
        if not words:
            return 0.0, 0.0, 0.0
        
        # 1. Numeric Density
        num_density = sum(1 for w in words if any(c.isdigit() for c in w)) / len(words) * 100
        
        # 2. Forward Looking Density
        # Keywords from NB2
        forward_terms = {
            "will", "expect", "expects", "expected", "expecting",
            "anticipate", "anticipates", "anticipated",
            "forecast", "forecasts", "forecasted",
            "project", "projects", "projected", "projecting",
            "estimate", "estimates", "estimated", "estimating",
            "intend", "intends", "intended",
            "plan", "plans", "planned", "planning",
            "believe", "believes", "believed",
            "outlook", "guidance", "target", "targets", "goal", "goals",
            "upcoming", "future", "forward",
            "should", "could", "may", "might", "would",
            "projected", "pending", "approximately"
        }
        multi_word_phrases = ["next quarter", "next year", "going forward", "full-year", "full year", "in the future"]
        
        text_lower = text.lower()
        fl_count = sum(1 for w in words if w.strip(".,;:!?\"'()[]").lower() in forward_terms)
        for phrase in multi_word_phrases:
            fl_count += text_lower.count(phrase)
        fl_density = fl_count / len(words) * 100
        
        # 3. Financial Symbol Density
        fs_density = (text.count("$") + text.count("%")) / len(words) * 100
        
        return num_density, fl_density, fs_density
    except Exception as e:
        logger.error(f"Failed to compute structural features: {e}")
        return 0.0, 0.0, 0.0

def llm_analyze_filing(text, model="gpt-4o"):
    """
    Uses OpenAI API to score the filing and extract key signals (Stage A3)
    as implemented in NB2_Event_Classification.
    """
    if not config.OPENAI_API_KEY:
        logger.error("OpenAI API Key is missing from configuration.")
        return None, ["API Key missing. Please set OPENAI_API_KEY in .env file."], "API Key missing. Cannot run LLM analysis."
    
    logger.info(f"Triggering OpenAI analysis (model: {model})")
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    truncated = text[:15000]
    prompt = USER_PROMPT.format(clean_text=truncated)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        import json
        data = json.loads(content)
        logger.info(f"LLM successfully analyzed filing. Importance: {data.get('importance_score')}")
        return data.get("importance_score"), data.get("key_signals", []), data.get("reasoning", "")
    except Exception as e:
        logger.error(f"Error during OpenAI API call: {str(e)}")
        return None, [f"Error: {str(e)}"], f"LLM Error: {str(e)}"

def calculate_mos_prospective(features, calibration_df):
    """
    Calculates the MOS score using calibration lift ratios (Stage C3).
    """
    try:
        if calibration_df is None or calibration_df.empty:
            logger.warning("Calibration table missing. returning neutral score.")
            return 0.5
        
        # Build lookup from calibration table
        lookup = {}
        for _, row in calibration_df.iterrows():
            feat = row['feature']
            val = str(row['value'])
            lift = row['lift'] if not pd.isna(row['lift']) else 1.0
            if feat not in lookup: lookup[feat] = {}
            lookup[feat][val] = lift
            
        score = 1.0
        
        # 1. Numeric Density
        nd = features.get('nd', 0)
        nd_tier = 'mid'
        if nd <= 6.476: nd_tier = 'low'
        elif nd >= 10.256: nd_tier = 'high'
        score *= lookup.get('numeric_density_tier', {}).get(nd_tier, 1.0)
        
        # 2. Forward Looking
        fld = features.get('fld', 0)
        fld_tier = 'mid'
        if fld <= 0.000: fld_tier = 'low'
        elif fld >= 1.234: fld_tier = 'high'
        score *= lookup.get('forward_looking_tier', {}).get(fld_tier, 1.0)
        
        # Normalize score
        final_mos = min(max(score / 5.0, 0.0), 1.0)
        return final_mos
    except Exception as e:
        logger.error(f"Error calculating MOS: {e}")
        return 0.5
