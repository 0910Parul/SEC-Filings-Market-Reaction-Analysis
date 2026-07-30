import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, backend

# Setup logging
logger = config.setup_logging("dashboard")
logger.info("Starting Dashboard...")

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="Not Yet Priced In",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# CSS — muted fintech aesthetic
# -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0D1117; }
section[data-testid="stSidebar"] { background-color: #0D1117; border-right: 1px solid #21262D; }

div[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 14px 18px;
}
div[data-testid="metric-container"] label {
    color: #8B949E !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #E6EDF3 !important;
    font-size: 1.8rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #161B22;
    border-radius: 8px;
    border: 1px solid #30363D;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #8B949E;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 6px 16px;
}
.stTabs [aria-selected="true"] {
    background: #21262D !important;
    color: #E6EDF3 !important;
}

div[data-testid="stExpander"] {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
}

p, li { color: #8B949E; font-size: 0.85rem; }
h1, h2, h3 { color: #E6EDF3; }

.section-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
COLORS = {
    'Delayed':     '#3FB950',
    'Gradual':     '#D29922',
    'Immediate':   '#388BFD',
    'No Reaction': '#6E7681',
}

PLOTLY_THEME = dict(
    paper_bgcolor='#0D1117',
    plot_bgcolor='#161B22',
    font=dict(family='Inter', color='#8B949E', size=11),
    xaxis=dict(gridcolor='#21262D', linecolor='#30363D', tickcolor='#30363D'),
    yaxis=dict(gridcolor='#21262D', linecolor='#30363D', tickcolor='#30363D'),
)

# -------------------------------------------------------
# Plug and play data loader
# -------------------------------------------------------
@st.cache_data
def load_data(master_file=None, watchlist_file=None, calib_file=None):
    try:
        logger.info("Attempting to load data...")
        result = {}

        # --- Load master (2023) ---
        frames = []
        if master_file:
            logger.info("Reading manual upload: master_scored.csv")
            df_master = pd.read_csv(master_file)
            df_master['year'] = 2023
            frames.append(df_master)
        elif config.MASTER_SCORED_CSV.exists():
            logger.info(f"Reading auto-discover: {config.MASTER_SCORED_CSV}")
            df_master = pd.read_csv(config.MASTER_SCORED_CSV)
            df_master['year'] = 2023
            frames.append(df_master)
        else:
            logger.warning(f"Master file not found at {config.MASTER_SCORED_CSV}")

        # --- Auto-discover backfill years (2020-2022) ---
        for year_dir in sorted(config.DATA_DIR.iterdir()):
            if year_dir.is_dir() and year_dir.name.isdigit():
                labelled = year_dir / "backfill_labelled.csv"
                checkpoint = year_dir / "backfill_checkpoint.csv"
                target = labelled if labelled.exists() else (checkpoint if checkpoint.exists() else None)
                if target:
                    logger.info(f"Merging backfill data: {target}")
                    df_bf = pd.read_csv(target)
                    if 'year' not in df_bf.columns:
                        df_bf['year'] = int(year_dir.name)
                    # Harmonize column names with master_scored
                    if 'company' in df_bf.columns and 'company_name' not in df_bf.columns:
                        df_bf.rename(columns={'company': 'company_name'}, inplace=True)
                    frames.append(df_bf)

        if frames:
            result['master'] = pd.concat(frames, ignore_index=True, sort=False)
            logger.info(f"Combined dataset: {len(result['master'])} rows across {result['master']['year'].nunique()} years")
        else:
            result['master'] = pd.DataFrame()

        if watchlist_file:
            result['watchlist'] = pd.read_csv(watchlist_file)
        elif config.WATCHLIST_FINAL_CSV.exists():
            result['watchlist'] = pd.read_csv(config.WATCHLIST_FINAL_CSV)
        else:
            result['watchlist'] = pd.DataFrame()

        if calib_file:
            result['calib'] = pd.read_csv(calib_file)
        elif config.CALIBRATION_TABLE_CSV.exists():
            result['calib'] = pd.read_csv(config.CALIBRATION_TABLE_CSV)
        else:
            result['calib'] = pd.DataFrame()

        for key in ['master', 'watchlist']:
            if not result[key].empty and 'filed_date' in result[key].columns:
                result[key]['filed_date'] = pd.to_datetime(result[key]['filed_date'], errors='coerce')

        return result
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Failed to load data files: {e}")
        return {'master': pd.DataFrame(), 'watchlist': pd.DataFrame(), 'calib': pd.DataFrame()}

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 Not Yet Priced In")
    st.markdown('<p style="font-size:0.72rem;color:#6E7681;margin-top:-8px;">AI Copilot · Delayed Reaction Detector</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-label">Data Source</p>', unsafe_allow_html=True)
    upload_mode = st.toggle("Upload CSV files manually", value=False)

    master_file = watchlist_file = calib_file = None
    if upload_mode:
        master_file    = st.file_uploader("master_scored.csv",     type="csv")
        watchlist_file = st.file_uploader("watchlist_final.csv",   type="csv")
        calib_file     = st.file_uploader("calibration_table.csv", type="csv")
    else:
        st.caption("Auto-reading from `data/` folder.")

    st.markdown("---")

    data      = load_data(master_file, watchlist_file, calib_file)
    master    = data.get('master',    pd.DataFrame())
    watchlist = data.get('watchlist', pd.DataFrame())
    calib     = data.get('calib',     pd.DataFrame())

    if master.empty:
        st.error("No data found. Add CSVs to `data/` folder or upload above.")
        st.stop()

    # --- On-the-fly MOS calculation for historical data missing it ---
    if not master.empty:
        # Ensure year column is present
        if 'year' not in master.columns:
            master['year'] = 2023
            
        # Calculate llm_importance_norm if missing
        if 'importance_score' in master.columns and 'llm_importance_norm' not in master.columns:
            master['llm_importance_norm'] = (master['importance_score'] - 1) / 4.0
            
        # Calculate reaction_pattern_score and MOS_retrospective if missing
        if 'MOS_retrospective' not in master.columns or master['MOS_retrospective'].isna().any():
            REACTION_SCORE_MAP = {'Delayed': 1.0, 'Gradual': 0.6, 'Immediate': 0.3, 'No Reaction': 0.1, 'Unknown': 0.1}
            
            # Fill missing importance norm with median
            if 'llm_importance_norm' in master.columns:
                master['llm_importance_norm'] = master['llm_importance_norm'].fillna(master['llm_importance_norm'].median())
            
            # Compute MOS for rows missing it
            mask = master['MOS_retrospective'].isna() if 'MOS_retrospective' in master.columns else pd.Series(True, index=master.index)
            if mask.any():
                pattern_score = master['reaction_class'].map(REACTION_SCORE_MAP).fillna(0.1)
                master.loc[mask, 'MOS_retrospective'] = (
                    0.4 * master.loc[mask, 'llm_importance_norm'].fillna(0.5) +
                    0.2 * master.loc[mask, 'baseline_importance'].fillna(0.5) +
                    0.4 * pattern_score[mask]
                )

    st.markdown('<p class="section-label">Filters</p>', unsafe_allow_html=True)

    if 'year' in master.columns:
        years = sorted(master['year'].unique(), reverse=True)
        # Default to all years
        selected_years = st.multiselect("Year", years, default=years)
    else:
        selected_years = []

    tickers = sorted(master['ticker'].unique()) if 'ticker' in master.columns else []
    selected_tickers = st.multiselect("Company", tickers, default=[])

    if 'broad_category' in master.columns:
        categories = sorted(master['broad_category'].dropna().unique())
        selected_cats = st.multiselect("Event Category", categories, default=[])
    else:
        selected_cats = []

    if 'reaction_class' in master.columns:
        reactions = sorted(master['reaction_class'].dropna().unique())
        selected_reactions = st.multiselect("Reaction Type", reactions, default=[])
    else:
        selected_reactions = []

    if 'MOS_retrospective' in master.columns:
        mos_min = float(master['MOS_retrospective'].min())
        mos_max = float(master['MOS_retrospective'].max())
        mos_threshold = st.slider("Min MOS Score", 0.0, 1.0,
            value=0.0, step=0.01, format="%.2f")
    else:
        mos_threshold = 0.0

    st.markdown("---")
    st.caption("Team 10 · University of Minnesota")
    st.caption("Big Data Trends & Market Analysis")

# -------------------------------------------------------
# Apply filters
# -------------------------------------------------------
filtered = master.copy()
if selected_years:
    filtered = filtered[filtered['year'].isin(selected_years)]
if selected_tickers:
    filtered = filtered[filtered['ticker'].isin(selected_tickers)]
if selected_cats and 'broad_category' in filtered.columns:
    filtered = filtered[filtered['broad_category'].isin(selected_cats)]
if selected_reactions and 'reaction_class' in filtered.columns:
    filtered = filtered[filtered['reaction_class'].isin(selected_reactions)]
if 'MOS_retrospective' in filtered.columns:
    filtered = filtered[filtered['MOS_retrospective'] >= mos_threshold]

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown("""
<div style="margin-bottom:1.5rem;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:500;color:#E6EDF3;">
        Not Yet Priced In
    </span>
    <span style="font-size:0.75rem;color:#8B949E;margin-left:12px;letter-spacing:1.5px;text-transform:uppercase;">
        AI Copilot · Delayed Market Reaction Detector
    </span>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Metric cards
# -------------------------------------------------------
total   = len(filtered)
delayed = len(filtered[filtered['reaction_class'] == 'Delayed']) if 'reaction_class' in filtered.columns else 0
pct_del = f"{delayed/total*100:.0f}%" if total > 0 else "—"
top_mos = f"{filtered['MOS_retrospective'].max():.3f}" if 'MOS_retrospective' in filtered.columns and not filtered.empty else "—"
avg_mos = f"{filtered['MOS_retrospective'].mean():.3f}" if 'MOS_retrospective' in filtered.columns and not filtered.empty else "—"

c1, c2, c3, c4, c5 = st.columns(5)
year_label = f"{len(selected_years)} Years" if len(selected_years) < len(years) else "All Data"
c1.metric("Total Filings",     total,     year_label)
c2.metric("Delayed Reactions", delayed,   pct_del)
c3.metric("Top MOS Score",     top_mos,   "Current Selection")
c4.metric("Avg MOS Score",     avg_mos)
c5.metric("Total Companies",   filtered['ticker'].nunique() if 'ticker' in filtered.columns else 0)

st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

# -------------------------------------------------------
# Tabs
# -------------------------------------------------------
tab_labels = [
    "📊 Overview",
    "🔬 Scatter Analysis",
    "🏆 Watchlist",
    "🧬 Live Analyzer",
    "🏗️ System Design",
    "⚙️ Backfill",
]

if config.SHOW_CALIBRATION:
    tab_labels.append("⚖️ Calibration")

all_tabs = st.tabs(tab_labels)

# Map tabs to variables for easier logic (handle dynamic count)
tab_overview = all_tabs[0]
tab_scatter  = all_tabs[1]
tab_watchlist = all_tabs[2]
tab_live      = all_tabs[3]
tab_design    = all_tabs[4]
tab_admin     = all_tabs[5]
tab_calib     = all_tabs[6] if config.SHOW_CALIBRATION else None

# ════════════════════════════════
# TAB: SYSTEM DESIGN
# ════════════════════════════════
with tab_design:
    st.markdown('<p class="section-label">Technical Architecture & Data Pipeline</p>', unsafe_allow_html=True)
    
    col_img, col_txt = st.columns([1, 1])
    
    with col_img:
        arch_path = config.REPO_ROOT / "assets" / "architecture.png"
        if arch_path.exists():
            st.image(str(arch_path), use_container_width=True)
        else:
            st.info("Architecture diagram asset not found. Run the generation script to create it.")
            
    with col_txt:
        st.markdown("""
        ### 🧪 Ground Truth Pipeline
        The "Oracle" of the system. It uses **CAR (Cumulative Abnormal Returns)** windows to identify 
        actual market movement. This data is then used to:
        *   **Label**: Categorize filings into Immediate, Delayed, or Gradual.
        *   **Score**: Compute the **MOS (Monetization Opportunity Score)** ground truth.
        *   **Calibrate**: Generate lift tables used for prospective scoring.
        
        ### 🏗️ Processing & Storage
        1.  **Parse & Extract**: Uses `parse_filing.py` to pull SEC categories, text features, and LLM (OpenAI/Gemini) importance scores.
        2.  **SQLite**: Stores structured metadata, audit trails, and ML features for relational analysis.
        3.  **ChromaDB**: Houses vector embeddings to power **Similarity Search**.
        
        ### 🤖 Intelligence Layer
        *   **ML Classifier**: Uses **XGBoost/Random Forest** models to predict the probability of a delayed reaction `P(Delayed)`.
        *   **RAG Retrieval**: Performs semantic lookups for the top-3 most similar historical filings to provide context boosts and MOS lift.
        """)

    st.markdown("---")
    st.markdown("""
    #### 🧬 Core Components
    - **Backfill Engine**: Processes historical years (2020-2022) to build a robust training set.
    - **LLM Scorer**: Provider-agnostic engine supporting OpenAI GPT-4o and Google Gemini for deep semantic analysis.
    - **Market Classifier**: Calculates cumulative returns across multi-day windows to identify reaction patterns.
    """)

# ════════════════════════════════
# TAB: WATCHLIST
# ════════════════════════════════
with tab_watchlist:
    st.markdown('<p class="section-label">Filings Ranked by MOS Retrospective</p>', unsafe_allow_html=True)

    # Use 'filtered' instead of 'watchlist' to allow showing all 900+ filings
    if filtered.empty:
        st.info("No filings match the current filters.")
    else:
        # We'll use the 'filtered' dataframe but ensure it's sorted by MOS
        if 'MOS_retrospective' in filtered.columns:
            filtered_wl = filtered.sort_values('MOS_retrospective', ascending=False)
            # Add a rank column if missing
            filtered_wl['rank'] = range(1, len(filtered_wl) + 1)
        else:
            filtered_wl = filtered.copy()
            filtered_wl['rank'] = range(1, len(filtered_wl) + 1)

        display_cols = [c for c in ['rank','ticker','filed_date','company_name',
            'broad_category','reaction_class','MOS_retrospective',
            'importance_score','R_short_0_1','R_long_5_20'] if c in filtered_wl.columns]

        wl_display = filtered_wl[display_cols].copy()

        if 'filed_date' in wl_display.columns:
            wl_display['filed_date'] = wl_display['filed_date'].dt.strftime('%Y-%m-%d')
        if 'MOS_retrospective' in wl_display.columns:
            wl_display['MOS_retrospective'] = wl_display['MOS_retrospective'].round(3)
        if 'R_short_0_1' in wl_display.columns:
            wl_display['R_short_0_1'] = (wl_display['R_short_0_1'] * 100).round(2).astype(str) + '%'
        if 'R_long_5_20' in wl_display.columns:
            wl_display['R_long_5_20'] = (wl_display['R_long_5_20'] * 100).round(2).astype(str) + '%'

        st.dataframe(wl_display, use_container_width=True, height=340, hide_index=True)

        st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Case Studies — Top 5</p>', unsafe_allow_html=True)

        for _, row in filtered_wl.head(5).iterrows():
            mos   = row.get('MOS_retrospective', 0)
            react = row.get('reaction_class', '')
            rc    = COLORS.get(react, '#8B949E')

            with st.expander(f"#{int(row.get('rank',0))}  {row.get('ticker','')}  —  {row.get('company_name','')}  ·  MOS {mos:.3f}"):
                col_a, col_b = st.columns(2)

                with col_a:
                    rs = row.get('R_short_0_1', 0) * 100
                    rl = row.get('R_long_5_20',  0) * 100
                    st.markdown(f"""
                    <div style='background:#161B22;border:1px solid #30363D;border-radius:8px;padding:14px;'>
                        <p style='color:#8B949E;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Filing Details</p>
                        <p style='color:#E6EDF3;margin:4px 0;font-size:0.83rem;'><b>Filed:</b> {str(row.get('filed_date',''))[:10]}</p>
                        <p style='color:#E6EDF3;margin:4px 0;font-size:0.83rem;'><b>Category:</b> {row.get('broad_category','')}</p>
                        <p style='color:#E6EDF3;margin:4px 0;font-size:0.83rem;'><b>Reaction:</b> <span style='color:{rc};font-weight:600;'>{react}</span></p>
                        <p style='color:#E6EDF3;margin:4px 0;font-size:0.83rem;'><b>LLM Importance:</b> {row.get('importance_score','—')}/5</p>
                        <p style='color:#E6EDF3;margin:4px 0;font-size:0.83rem;'><b>MOS Score:</b> {mos:.3f}</p>
                        <hr style='border-color:#30363D;margin:10px 0;'>
                        <p style='color:#E6EDF3;margin:4px 0;font-size:0.83rem;'>
                            Day 0–1: <span style='color:{"#3FB950" if rs>=0 else "#F85149"};font-family:monospace;font-weight:600;'>{rs:+.2f}%</span>
                            &nbsp;&nbsp;
                            Day 5–20: <span style='color:{"#3FB950" if rl>=0 else "#F85149"};font-family:monospace;font-weight:600;'>{rl:+.2f}%</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_b:
                    reasoning = str(row.get('reasoning', ''))
                    st.markdown(f"""
                    <div style='background:#161B22;border:1px solid #30363D;border-radius:8px;padding:14px;'>
                        <p style='color:#8B949E;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>AI Reasoning</p>
                        <p style='color:#E6EDF3;font-size:0.81rem;line-height:1.6;'>{reasoning[:280]}{'...' if len(reasoning)>280 else ''}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    try:
                        signals = json.loads(row.get('key_signals','[]'))
                        if signals:
                            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
                            for sig in signals[:3]:
                                st.markdown(f"""
                                <div style='background:#0D1117;border-left:2px solid #238636;padding:6px 10px;margin-bottom:4px;border-radius:0 4px 4px 0;'>
                                    <p style='color:#8B949E;font-size:0.76rem;margin:0;font-style:italic;'>"{str(sig)[:120]}..."</p>
                                </div>
                                """, unsafe_allow_html=True)
                    except Exception:
                        pass

# ════════════════════════════════
# TAB: OVERVIEW
# ════════════════════════════════
with tab_overview:

    # ── Row 1: Pie + Bar side by side ──
    st.markdown('<p class="section-label">Reaction Class Distribution & Monetization Potential</p>', unsafe_allow_html=True)

    col_pie, col_bar = st.columns(2)

    with col_pie:
        if 'reaction_class' in filtered.columns:
            rc_df = filtered['reaction_class'].value_counts().reset_index()
            rc_df.columns = ['reaction_class', 'count']
            fig_pie = px.pie(rc_df, values='count', names='reaction_class',
                color='reaction_class', color_discrete_map=COLORS, hole=0.55)
            fig_pie.update_traces(
                textposition='outside', textinfo='label+percent',
                textfont_size=10,
                pull=[0.12 if r == 'Delayed' else 0 for r in rc_df['reaction_class']]
            )
            fig_pie.update_layout(**PLOTLY_THEME, showlegend=False,
                margin=dict(t=20,b=20,l=20,r=20), height=230)
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        if 'MOS_retrospective' in filtered.columns and 'reaction_class' in filtered.columns:
            mos_by_rc = filtered.groupby('reaction_class')['MOS_retrospective'].mean().reset_index()
            mos_by_rc.columns = ['reaction_class', 'avg_MOS']
            mos_by_rc = mos_by_rc[mos_by_rc['reaction_class'] != 'Unknown']
            mos_by_rc = mos_by_rc.sort_values('avg_MOS', ascending=True)
            mos_by_rc['color'] = mos_by_rc['reaction_class'].map(COLORS).fillna('#6E7681')
            fig_mos = go.Figure(go.Bar(
                x=mos_by_rc['avg_MOS'],
                y=mos_by_rc['reaction_class'],
                orientation='h',
                marker_color=mos_by_rc['color'],
                text=mos_by_rc['avg_MOS'].round(2),
                textposition='outside',
                textfont=dict(color='#8B949E', size=10),
            ))
            fig_mos.update_layout(
                **{k:v for k,v in PLOTLY_THEME.items() if k not in ['xaxis','yaxis']},
                xaxis_title="Avg MOS Score", yaxis_title="",
                showlegend=False,
                margin=dict(t=10,b=30,l=10,r=50), height=230,
                xaxis=dict(range=[0,1], gridcolor='#21262D', linecolor='#30363D', tickfont=dict(size=9)),
                yaxis=dict(gridcolor='#21262D', linecolor='#30363D', tickfont=dict(size=10))
            )
            st.plotly_chart(fig_mos, use_container_width=True)

    # ── Insight text + metrics below both charts ──
    if 'reaction_class' in filtered.columns and 'MOS_retrospective' in filtered.columns:
        delayed_df  = filtered[filtered['reaction_class'] == 'Delayed']
        imm_df      = filtered[filtered['reaction_class'] == 'Immediate']
        delayed_mos = delayed_df['MOS_retrospective'].mean() if not delayed_df.empty else 0
        imm_mos     = imm_df['MOS_retrospective'].mean() if not imm_df.empty else 1
        ratio       = delayed_mos / imm_mos if imm_mos > 0 else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Delayed filings found", len(delayed_df),        "missed opportunities")
        m2.metric("Avg MOS — delayed",     f"{delayed_mos:.2f}",   "highest of all classes")
        m3.metric("Potential multiplier",  f"{ratio:.1f}×",        "vs immediate reactions")

        st.markdown(f"""
        <p style='font-size:0.75rem;color:#6E7681;margin-top:4px;'>
        Delayed reactions are rare ({len(delayed_df)} out of {len(filtered)} filings) but carry
        the highest monetization potential — avg MOS of {delayed_mos:.2f} vs {imm_mos:.2f} for
        immediate reactions. These are filings where material information was public but
        <b style='color:#3FB950;'>not yet priced in</b>.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 2: MOS Distribution — full width ──
    st.markdown('<p class="section-label">MOS Score Distribution</p>', unsafe_allow_html=True)
    if 'MOS_retrospective' in filtered.columns:
        fig_hist = px.histogram(filtered, x='MOS_retrospective', nbins=30,
            color_discrete_sequence=['#238636'])
        p90 = filtered['MOS_retrospective'].quantile(0.9)
        fig_hist.add_vline(x=p90, line_dash="dash", line_color="#D29922",
            annotation_text=f"Top 10% ({p90:.2f})",
            annotation_font_color="#D29922", annotation_font_size=10)
        fig_hist.update_layout(**PLOTLY_THEME,
            xaxis_title="MOS Score", yaxis_title="Count",
            showlegend=False, bargap=0.05,
            margin=dict(t=20,b=40,l=40,r=20), height=200)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # ── Row 3: Filings by Category — full width ──
    st.markdown('<p class="section-label">Filings by Event Category</p>', unsafe_allow_html=True)
    if 'broad_category' in filtered.columns and 'reaction_class' in filtered.columns:
        cat_df = filtered[filtered['reaction_class'] != 'Unknown'].groupby(
            ['broad_category','reaction_class']).size().reset_index(name='count')
        fig_bar = px.bar(cat_df, x='broad_category', y='count',
            color='reaction_class', color_discrete_map=COLORS, barmode='stack')
        fig_bar.update_layout(**PLOTLY_THEME,
            xaxis_title="", yaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                font=dict(color='#8B949E', size=11)),
            margin=dict(t=30,b=80,l=40,r=20), height=320)
        fig_bar.update_xaxes(tickangle=20, tickfont=dict(size=10))
        st.plotly_chart(fig_bar, use_container_width=True)


# ════════════════════════════════
# TAB: CALIBRATION
# ════════════════════════════════
if tab_calib:
    with tab_calib:
        if calib.empty:
            st.info("calibration_table.csv not found in data/ folder.")
        else:
            st.markdown('<p class="section-label">Which Features Best Predict Delayed Reactions?</p>', unsafe_allow_html=True)
            st.caption("Lift = P(feature | Delayed) / P(feature | Non-Delayed). Values > 1 indicate the feature is more common in delayed-reaction filings.")

            c1, c2 = st.columns([1.2, 1])

            with c1:
                if 'lift' in calib.columns:
                    strong   = calib[calib['lift'] >= 1.7].head(8)
                    moderate = calib[(calib['lift'] >= 1.3) & (calib['lift'] < 1.7)].head(8)

                    st.markdown('<p class="section-label" style="color:#3FB950;">Strong Signals — Lift ≥ 1.7</p>', unsafe_allow_html=True)
                    if not strong.empty:
                        cols = [c for c in ['feature','value','lift'] if c in strong.columns]
                        display_strong = strong[cols].copy().round(2)
                        display_strong['feature'] = display_strong['feature'].replace({
                            'broad_category': 'Filing Type',
                            'baseline_importance_tier': 'Text Importance',
                            'llm_importance_tier': 'AI Importance',
                            'numeric_density_tier': 'Number Density',
                            'forward_looking_tier': 'Forward Looking',
                        })
                        st.dataframe(display_strong, use_container_width=True, hide_index=True)

                    st.markdown('<p class="section-label" style="margin-top:1rem;color:#D29922;">Moderate Signals — Lift 1.3–1.7</p>', unsafe_allow_html=True)
                    if not moderate.empty:
                        cols = [c for c in ['feature','value','lift'] if c in moderate.columns]
                        display_moderate = moderate[cols].copy().round(2)
                        display_moderate['feature'] = display_moderate['feature'].replace({
                            'broad_category': 'Filing Type',
                            'baseline_importance_tier': 'Text Importance',
                            'llm_importance_tier': 'AI Importance',
                            'numeric_density_tier': 'Number Density',
                            'forward_looking_tier': 'Forward Looking',
                        })
                        st.dataframe(display_moderate, use_container_width=True, hide_index=True)
            with c2:
                st.markdown('<p class="section-label">Top Lift Ratios</p>', unsafe_allow_html=True)
                if 'lift' in calib.columns:
                    top_c = calib.nlargest(12, 'lift').copy()
                    label_map = {
                        ('broad_category', 'Acquisition/Disposition'): 'Filing Type: Acquisition',
                        ('broad_category', 'Governance'):              'Filing Type: Governance',
                        ('broad_category', 'Regulatory'):              'Filing Type: Regulatory',
                        ('broad_category', 'Executive Change'):        'Filing Type: Executive Change',
                        ('broad_category', 'Other'):                   'Filing Type: Other',
                        ('broad_category', 'Earnings'):                'Filing Type: Earnings',
                        ('baseline_importance_tier', 'high'):          'Text Importance: High',
                        ('baseline_importance_tier', 'mid'):           'Text Importance: Medium',
                        ('baseline_importance_tier', 'low'):           'Text Importance: Low',
                        ('llm_importance_tier', 'high'):               'AI Importance: High',
                        ('llm_importance_tier', 'mid'):                'AI Importance: Medium',
                        ('llm_importance_tier', 'low'):                'AI Importance: Low',
                        ('numeric_density_tier', 'high'):              'Number Density: High',
                        ('numeric_density_tier', 'mid'):               'Number Density: Medium',
                        ('numeric_density_tier', 'low'):               'Number Density: Low',
                        ('forward_looking_tier', 'high'):              'Forward Looking: High',
                        ('forward_looking_tier', 'mid'):               'Forward Looking: Medium',
                        ('forward_looking_tier', 'low'):               'Forward Looking: Low',
                    }
                    top_c['label'] = top_c.apply(
                        lambda r: label_map.get((r['feature'], r['value']),
                        r['feature'] + ': ' + str(r['value'])), axis=1)
                    top_c['color'] = top_c['lift'].apply(
                        lambda x: '#3FB950' if x >= 1.7 else ('#D29922' if x >= 1.3 else '#6E7681'))
                    fig_c = go.Figure(go.Bar(
                        x=top_c['lift'], y=top_c['label'], orientation='h',
                        marker_color=top_c['color'],
                        text=top_c['lift'].round(2), textposition='outside',
                        textfont=dict(color='#8B949E', size=10),
                    ))
                    fig_c.add_vline(x=1.0, line_dash="dash", line_color="#30363D")
                    fig_c.update_layout(**{k:v for k,v in PLOTLY_THEME.items() if k != 'yaxis'},
                        xaxis_title="Lift Ratio", yaxis_title="",
                        margin=dict(t=10,b=40,l=10,r=60), height=400,
                        yaxis=dict(gridcolor='#21262D', linecolor='#30363D',
                            tickcolor='#30363D', tickfont=dict(size=9)))
                    st.plotly_chart(fig_c, use_container_width=True)

# ════════════════════════════════
# TAB: SCATTER
# ════════════════════════════════
with tab_scatter:
    if master.empty or 'R_short_0_1' not in master.columns:
        st.info("Return columns not found.")
    else:
        st.markdown('<p class="section-label">Immediate vs Delayed Reaction — Each dot is one filing</p>', unsafe_allow_html=True)
        st.caption("Top-left quadrant = Not Yet Priced In zone — small immediate reaction but significant delayed move.")

        plot_df = filtered.copy() if not filtered.empty else master.copy()
        plot_df['r_short_pct'] = plot_df['R_short_0_1'] * 100
        plot_df['r_long_pct']  = plot_df['R_long_5_20']  * 100
        plot_df['hover_name']  = plot_df.get('ticker', '').astype(str) + ' — ' + plot_df.get('company_name', '').astype(str)

        fig_s = px.scatter(plot_df, x='r_short_pct', y='r_long_pct',
            color='reaction_class', color_discrete_map=COLORS,
            hover_name='hover_name',
            hover_data={'r_short_pct':':.2f','r_long_pct':':.2f',
                'MOS_retrospective':':.3f','broad_category':True,'reaction_class':False},
            opacity=0.75)

        fig_s.add_hline(y=0,  line_dash="dot", line_color="#30363D", line_width=1)
        fig_s.add_vline(x=0,  line_dash="dot", line_color="#30363D", line_width=1)
        fig_s.add_shape(type="rect", x0=-20, x1=2, y0=2, y1=35,
            fillcolor="rgba(35,134,54,0.05)",
            line=dict(color="rgba(35,134,54,0.25)", width=0.8, dash="dot"))
        fig_s.add_annotation(x=-10, y=33, text="Not Yet Priced In Zone",
            font=dict(color="#238636", size=10), showarrow=False)

        fig_s.update_layout(**PLOTLY_THEME,
            xaxis_title="Day 0–1 Return (%)", yaxis_title="Day 5–20 Return (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                font=dict(color='#8B949E', size=11)),
            margin=dict(t=40,b=60,l=60,r=20), height=500)
        fig_s.update_traces(marker=dict(size=7))
        st.plotly_chart(fig_s, use_container_width=True)

        if 'reaction_class' in plot_df.columns:
            st.markdown('<p class="section-label" style="margin-top:1rem;">Summary by Reaction Class</p>', unsafe_allow_html=True)
            summary = plot_df.groupby('reaction_class').agg(
                Count=('ticker','count'),
                Avg_Day01=('r_short_pct','mean'),
                Avg_Day520=('r_long_pct','mean'),
                Avg_MOS=('MOS_retrospective','mean'),
            ).round(3).reset_index()
            st.dataframe(summary, use_container_width=True, hide_index=True)

# ════════════════════════════════
# TAB: LIVE ANALYZER
# ════════════════════════════════
with tab_live:
    st.markdown('<p class="section-label">Real-Time Filing Analysis (Stage A Pipeline)</p>', unsafe_allow_html=True)
    st.caption("Paste raw 8-K text below to run the parser, feature extractor, and AI scorer in real-time.")

    raw_text = st.text_area("SEC 8-K Raw Text", height=300, placeholder="Paste filing content here...")
    
    if st.button("🚀 Analyze Filing", use_container_width=True):
        if not raw_text:
            st.warning("Please paste some text first.")
        else:
            logger.info("User triggered 'Analyze Filing' pipeline.")
            with st.spinner("Parsing and scoring..."):
                try:
                    # 1. Parse
                    logger.info("Step 1: Parsing raw text")
                    parsed_result = backend.parse_filing(raw_text)
                    
                    # Extract string if dict, else assume string
                    clean_text = parsed_result.get("clean_text", "") if isinstance(parsed_result, dict) else parsed_result
                    
                    if not clean_text or len(clean_text) < 10:
                        st.error("Parser failed to extract meaningful text from the filing. Please check the format.")
                        st.stop()

                    # 2. Structural Features
                    logger.info("Step 2: Computing structural features")
                    nd, fld, fsd = backend.compute_structural_features(clean_text)
                    
                    # 3. LLM Score (A3)
                    logger.info("Step 3: Calling OpenAI analyzer")
                    score, signals, reasoning = backend.llm_analyze_filing(clean_text)
                    
                    # 4. Final MOS (Prospective)
                    logger.info("Step 4: Calculating MOS score")
                    mos_p = backend.calculate_mos_prospective({'nd': nd, 'fld': fld, 'fsd': fsd}, calib)

                    logger.info("Pipepline complete. Rendering results.")
                    st.markdown("---")
                except Exception as e:
                    logger.error(f"Pipeline error: {e}")
                    st.error(f"Analysis Failed: {e}")
                    st.stop()
                
                # Layout Results
                res_c1, res_c2 = st.columns([1, 2])
                
                with res_c1:
                    st.metric("Predicted MOS", f"{mos_p:.3f}", "Prospective Score")
                    st.metric("AI Importance", f"{score}/5" if score else "—")
                    
                    # Mini Gauge
                    fig_g = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = mos_p,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence", 'font': {'size': 14}},
                        gauge = {
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "#3FB950"},
                            'steps' : [
                                {'range': [0, 0.4], 'color': "#21262D"},
                                {'range': [0.4, 0.7], 'color': "#30363D"}
                            ],
                        }
                    ))
                    fig_g.update_layout(**PLOTLY_THEME, height=200, margin=dict(t=30, b=20, l=20, r=20))
                    st.plotly_chart(fig_g, use_container_width=True)

                with res_c2:
                    st.markdown(f"### 🤖 AI Reasoning")
                    st.info(reasoning)
                    
                    st.markdown("### 📡 Key Signals")
                    if isinstance(signals, list):
                        for sig in signals:
                            st.markdown(f"- {sig}")
                    else:
                        st.write(signals)
                
                st.markdown("---")
                st.markdown("### 📊 Structural Indicators")
                f_c1, f_c2, f_c3 = st.columns(3)
                f_c1.metric("Numeric Density", f"{nd:.2f}%")
                f_c2.metric("Forward Looking", f"{fld:.2f}%")
                f_c3.metric("Financial Symbols", f"{fsd:.2f}%")

# ════════════════════════════════
# TAB: ADMIN / BACKFILL
# ════════════════════════════════
with tab_admin:
    st.markdown('<p class="section-label">Historical Backfill Orchestrator</p>', unsafe_allow_html=True)
    st.caption("Process historical years (2020-2022) to rebuild the training dataset using OpenAI GPT-4o.")

    import subprocess
    import signal
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        years_to_run = st.multiselect("Select Years to Backfill", [2020, 2021, 2022], default=[2022])
        skip_llm = st.checkbox("Skip LLM Stage (Features Only)", value=False)
        
    with col2:
        if "backfill_proc" not in st.session_state:
            st.session_state.backfill_proc = None
            
        if st.session_state.backfill_proc is None or st.session_state.backfill_proc.poll() is not None:
            if st.button("▶️ Start Backfill", use_container_width=True):
                years_str = " ".join(map(str, years_to_run))
                cmd = [f"{config.REPO_ROOT}/.venv/bin/python", f"{config.REPO_ROOT}/src/backfill.py", "--years"] + years_str.split()
                if skip_llm: cmd.append("--skip-llm")
                
                try:
                    # Clear session state if needed or just start
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    st.session_state.backfill_proc = proc
                    st.success(f"Backfill started for years: {years_str}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start process: {e}")
        else:
            if st.button("⏹️ Stop Backfill", use_container_width=True, type="primary"):
                st.session_state.backfill_proc.terminate()
                st.session_state.backfill_proc = None
                st.warning("Backfill process terminated.")
                st.rerun()

    # Log Viewer
    st.markdown("---")
    st.markdown('<p class="section-label">Process Logs (Latest app.log Entries)</p>', unsafe_allow_html=True)
    
    try:
        log_path = config.LOG_DIR / "app.log"
        if log_path.exists():
            with open(log_path, "r") as f:
                lines = f.readlines()[-20:]
                st.code("".join(lines))
        else:
            st.info("Log file not found yet.")
    except Exception as e:
        st.write(f"Could not read logs: {e}")

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center;'>
    <p style='font-size:0.7rem;color:#6E7681;'>
        Team 10 · Big Data Trends & Market Analysis · University of Minnesota ·
        <a href='https://github.com/ShivanshuDagur/not-yet-priced-in' style='color:#238636;text-decoration:none;'>
            github.com/ShivanshuDagur/not-yet-priced-in
        </a>
    </p>
    <p style='font-size:0.65rem;color:#6E7681;'>
        For research and educational purposes only. Does not constitute financial advice.
    </p>
</div>
""", unsafe_allow_html=True)
