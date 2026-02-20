# ==========================
# main.py - OTT Analytics RAG (No API Key Required)
# ==========================
import streamlit as st
import faiss, pickle, json, os, re
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter, defaultdict
import pandas as pd

INDEX_PATH = "vector.index"
META_PATH  = "metadata.pkl"
STATS_PATH = "dataset_stats.json"
CSV_PATH   = "ott_users_2016_2026.csv"

st.set_page_config(page_title="OTT Pulse · Analytics RAG", page_icon="📡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
:root{--bg:#080c14;--surface:#0e1623;--border:#1c2a40;--accent:#00e5ff;--accent2:#7b61ff;--accent3:#ff6b6b;--text:#d4e4f7;--muted:#5a7a9a;--glow:0 0 24px rgba(0,229,255,0.18)}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif!important}
#MainMenu,footer,header{visibility:hidden}[data-testid="stDecoration"]{display:none}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important}
[data-testid="stSidebar"] *{color:var(--text)!important}
.ott-hero{text-align:center;padding:2.5rem 1rem 1.5rem}
.ott-hero h1{font-family:'Syne',sans-serif;font-size:clamp(2.2rem,5vw,3.8rem);font-weight:800;background:linear-gradient(135deg,#00e5ff 0%,#7b61ff 50%,#ff6b6b 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0;letter-spacing:-0.02em}
.ott-hero p{color:var(--muted);font-size:1rem;margin-top:.4rem;font-family:'DM Mono',monospace;letter-spacing:.05em}
.stTextInput>div>div>input{background:var(--surface)!important;border:1.5px solid var(--border)!important;color:var(--text)!important;border-radius:12px!important;font-family:'DM Sans',sans-serif!important;font-size:1rem!important;padding:.75rem 1rem!important;transition:border-color .2s,box-shadow .2s!important}
.stTextInput>div>div>input:focus{border-color:var(--accent)!important;box-shadow:var(--glow)!important}
.stTextInput label{color:var(--muted)!important;font-size:.8rem!important}
.stButton>button{background:linear-gradient(135deg,#00e5ff,#7b61ff)!important;color:#080c14!important;border:none!important;border-radius:10px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:.95rem!important;padding:.6rem 1.8rem!important;transition:opacity .2s,transform .15s!important}
.stButton>button:hover{opacity:.88!important;transform:translateY(-1px)!important}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem 1.2rem;text-align:center}
.stat-card .val{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:var(--accent)}
.stat-card .lbl{font-size:.72rem;color:var(--muted);margin-top:.2rem;letter-spacing:.06em;text-transform:uppercase}
.synthesis-box{background:linear-gradient(135deg,rgba(0,229,255,.06),rgba(123,97,255,.06));border:1px solid rgba(0,229,255,.25);border-radius:16px;padding:1.5rem 1.8rem;margin:1.2rem 0;box-shadow:var(--glow)}
.synthesis-box .label{font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.12em;color:var(--accent);margin-bottom:.8rem;display:block}
.insight-row{display:flex;align-items:flex-start;gap:.6rem;margin:.55rem 0;font-size:.93rem;line-height:1.65}
.insight-bullet{color:var(--accent);font-weight:700;flex-shrink:0;margin-top:.05rem}
.result-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem 1.2rem;margin:.5rem 0;font-family:'DM Mono',monospace;font-size:.78rem;line-height:1.8;color:var(--muted);transition:border-color .2s}
.result-card:hover{border-color:var(--accent2)}
.result-card .rank{font-family:'Syne',sans-serif;font-weight:700;font-size:.7rem;color:var(--accent);letter-spacing:.1em}
.score-tag{display:inline-block;background:rgba(123,97,255,.15);border:1px solid rgba(123,97,255,.3);border-radius:6px;padding:.1rem .5rem;font-size:.68rem;font-family:'DM Mono',monospace;color:#a895ff;float:right}
.section-title{font-family:'Syne',sans-serif;font-size:.75rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:1.5rem 0 .6rem;padding-bottom:.4rem;border-bottom:1px solid var(--border)}
.chip{display:inline-block;background:rgba(0,229,255,.1);border:1px solid rgba(0,229,255,.2);border-radius:20px;padding:.2rem .7rem;font-size:.7rem;font-family:'DM Mono',monospace;color:var(--accent);margin:.2rem .15rem}
.stSelectbox>div>div{background:var(--surface)!important;border-color:var(--border)!important;border-radius:10px!important;color:var(--text)!important}
.stSlider label{color:var(--muted)!important;font-size:.8rem!important}
.streamlit-expanderHeader{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text)!important;font-family:'DM Mono',monospace!important;font-size:.82rem!important}
hr{border:none;border-top:1px solid var(--border)!important;margin:1.5rem 0}
</style>""", unsafe_allow_html=True)


# ── Loaders ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    if isinstance(meta, list):
        texts, records = meta, None
    else:
        texts   = meta.get("texts", [])
        records = meta.get("records", None)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, texts, records, model

@st.cache_data(show_spinner=False)
def load_stats():
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            return json.load(f)
    return None

@st.cache_data(show_spinner=False)
def load_df():
    """Load the full CSV — used for accurate aggregate analytics."""
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return None


# ── Search ───────────────────────────────────────────────────────────────────────
def search(query: str, k: int = 8):
    index, texts, records, model = load_resources()
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "text":   texts[idx],
                "record": records[idx] if records else None,
                "score":  float(score),
            })
    return results


# ── Smart Local Synthesizer (uses FULL dataset for accuracy) ──────────────────
def pick_best_metric(query: str, num_cols: list) -> str:
    """
    Match query keywords to the most relevant numeric column.
    Falls back to the first column if nothing matches.
    """
    q = query.lower()
    keyword_map = {
        "revenue":      ["revenue", "income", "earn", "sales"],
        "churn":        ["churn", "cancel", "attrition", "lost"],
        "user":         ["user", "subscriber", "member", "account"],
        "watch":        ["watch", "view", "hour", "time", "duration"],
        "growth":       ["growth", "growth_rate", "increase"],
        "rating":       ["rating", "score", "satisfaction"],
        "content":      ["content", "title", "show", "genre"],
        "subscription": ["subscription", "plan", "tier"],
    }
    for col in num_cols:
        col_lower = col.lower()
        for intent, keywords in keyword_map.items():
            if any(kw in col_lower for kw in keywords):
                if any(kw in q for kw in keywords):
                    return col
    # fallback: first column whose name appears in query
    for col in num_cols:
        if any(part in q for part in col.lower().split("_")):
            return col
    return num_cols[0]


def synthesize(query: str, results: list) -> list:
    """
    Accurate intent-aware synthesizer.
    FIX: Runs analytics on the FULL dataset, not just the retrieved k records.
    Picks the most query-relevant numeric column instead of always using index 0.
    """

    q = query.lower()

    # ── Use the full CSV for accurate stats ──────────────────────────────────
    full_df = load_df()

    # Fallback: if CSV not found, use retrieved records (less accurate)
    if full_df is None or full_df.empty:
        records = [r["record"] for r in results if r["record"]]
        if not records:
            return [f"Found {len(results)} semantic matches. No structured records available."]
        full_df = pd.DataFrame(records)
        data_source_note = f"⚠️ Full CSV not found — stats computed from {len(records)} retrieved records only."
    else:
        data_source_note = f"Stats computed from full dataset ({len(full_df):,} records)."

    df = full_df.copy()
    insights = [data_source_note]

    # ── Intent Detection ──────────────────────────────────────────────────────
    is_trend   = any(w in q for w in ["trend", "over time", "by year", "growth", "year"])
    is_top     = any(w in q for w in ["top", "highest", "best", "leading", "most"])
    is_compare = any(w in q for w in ["vs", "compare", "versus", "difference"])
    is_churn   = any(w in q for w in ["churn", "cancel", "attrition"])
    is_region  = any(w in q for w in ["region", "country", "location", "regional"])

    # ── Apply Year Filter (if specific years mentioned) ───────────────────────
    year_col = next((c for c in df.columns if "year" in c.lower()), None)
    years_in_query = [int(y) for y in re.findall(r"\b(20\d{2})\b", q)]
    if year_col and years_in_query:
        df = df[df[year_col].isin(years_in_query)]
        insights.append(
            f"Filtered to **{len(df):,} records** for year(s): {', '.join(map(str, years_in_query))}."
        )
        if df.empty:
            return insights + ["No records found for the specified year(s)."]

    # ── Column Classification ─────────────────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    # Exclude year column from metrics
    if year_col and year_col in num_cols:
        num_cols = [c for c in num_cols if c != year_col]
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    # Also exclude year from cat if it's object type
    cat_cols = [c for c in cat_cols if "year" not in c.lower() or c == year_col]

    if not num_cols:
        return insights + ["No numeric columns available for analytical insights."]

    # ── Pick Most Relevant Metric ─────────────────────────────────────────────
    primary_metric = pick_best_metric(query, num_cols)

    # ── Find Best Categorical Grouping Column ─────────────────────────────────
    def best_cat_col(preferred_keywords: list) -> str | None:
        for kw in preferred_keywords:
            for c in cat_cols:
                if kw in c.lower():
                    return c
        return cat_cols[0] if cat_cols else None

    # ─────────────────────────────────────────────────────────────────────────
    # 1️⃣ CHURN INTENT
    # ─────────────────────────────────────────────────────────────────────────
    if is_churn:
        churn_col = next((c for c in num_cols if "churn" in c.lower()), primary_metric)
        total_churn = df[churn_col].sum()
        avg_churn   = df[churn_col].mean()
        insights.append(f"**Churn metric:** `{churn_col}`")
        insights.append(f"Total churn sum: **{total_churn:,.2f}** | Average: **{avg_churn:,.2f}**")

        # Churn by platform/category
        group_col = best_cat_col(["platform", "service", "provider"])
        if group_col:
            ranked = df.groupby(group_col)[churn_col].mean().sort_values(ascending=False)
            insights.append(f"Highest average churn by **{group_col}**:")
            for i, (name, val) in enumerate(ranked.head(5).items()):
                insights.append(f"{i+1}. **{name}** → {val:,.2f}")

        # Churn trend by year
        if year_col and not years_in_query:
            trend = df.groupby(year_col)[churn_col].mean().sort_index()
            if len(trend) > 1:
                direction = "↑ increasing" if trend.iloc[-1] > trend.iloc[0] else "↓ decreasing"
                change = ((trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0]) * 100
                insights.append(
                    f"Churn trend {int(trend.index.min())}–{int(trend.index.max())}: "
                    f"{direction} by **{abs(change):.1f}%**"
                )

    # ─────────────────────────────────────────────────────────────────────────
    # 2️⃣ TREND / YEAR INTENT
    # ─────────────────────────────────────────────────────────────────────────
    elif is_trend and year_col:
        trend = df.groupby(year_col)[primary_metric].sum().sort_index()
        insights.append(f"**{primary_metric}** by year (full dataset):")
        for yr, val in trend.items():
            insights.append(f"• {int(yr)}: **{val:,.2f}**")

        if len(trend) > 1:
            growth = ((trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0]) * 100
            direction = "📈 grew" if growth > 0 else "📉 declined"
            insights.append(
                f"Overall: {direction} by **{abs(growth):.1f}%** "
                f"({int(trend.index.min())} → {int(trend.index.max())})"
            )

        # YoY best/worst year
        if len(trend) > 1:
            yoy = trend.pct_change() * 100
            best_yr  = yoy.idxmax()
            worst_yr = yoy.idxmin()
            insights.append(
                f"Best YoY growth: **{int(best_yr)}** ({yoy[best_yr]:+.1f}%) | "
                f"Worst: **{int(worst_yr)}** ({yoy[worst_yr]:+.1f}%)"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 3️⃣ TOP / RANKING INTENT
    # ─────────────────────────────────────────────────────────────────────────
    elif is_top:
        group_col = best_cat_col(["platform", "service", "genre", "region", "country", "plan"])
        if group_col:
            grouped = df.groupby(group_col)[primary_metric].sum().sort_values(ascending=False)
            insights.append(f"Top entities ranked by **{primary_metric}** (from {len(df):,} records):")
            for i, (name, value) in enumerate(grouped.head(5).items()):
                share = (value / grouped.sum()) * 100
                insights.append(f"{i+1}. **{name}** → {value:,.2f} ({share:.1f}% share)")

            leader_share = (grouped.iloc[0] / grouped.sum()) * 100
            insights.append(
                f"Market concentration: Leader controls **{leader_share:.1f}%** of total {primary_metric}."
            )
        else:
            # No cat col — show top raw values
            top_vals = df[primary_metric].nlargest(5)
            insights.append(f"Top 5 values for **{primary_metric}**:")
            for i, v in enumerate(top_vals):
                insights.append(f"{i+1}. **{v:,.2f}**")

    # ─────────────────────────────────────────────────────────────────────────
    # 4️⃣ COMPARE INTENT
    # ─────────────────────────────────────────────────────────────────────────
    elif is_compare:
        group_col = best_cat_col(["platform", "service", "region", "plan", "genre"])
        if group_col:
            grouped = df.groupby(group_col)[primary_metric].agg(["mean", "sum", "count"])
            grouped.columns = ["avg", "total", "count"]
            grouped = grouped.sort_values("avg", ascending=False)
            insights.append(f"Comparison by **{group_col}** for `{primary_metric}`:")
            for name, row in grouped.head(6).iterrows():
                insights.append(
                    f"• **{name}** → avg {row['avg']:,.2f} | total {row['total']:,.2f} | n={int(row['count'])}"
                )
            diff = grouped["avg"].max() - grouped["avg"].min()
            insights.append(f"Performance gap (max − min avg): **{diff:,.2f}**")

    # ─────────────────────────────────────────────────────────────────────────
    # 5️⃣ REGIONAL INTENT
    # ─────────────────────────────────────────────────────────────────────────
    elif is_region:
        region_col = best_cat_col(["region", "country", "location", "area"])
        if region_col:
            grouped = df.groupby(region_col)[primary_metric].sum().sort_values(ascending=False)
            insights.append(f"**{primary_metric}** breakdown by **{region_col}**:")
            for i, (name, val) in enumerate(grouped.head(6).items()):
                share = (val / grouped.sum()) * 100
                insights.append(f"{i+1}. **{name}** → {val:,.2f} ({share:.1f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # 6️⃣ DEFAULT: Full Analytical Summary
    # ─────────────────────────────────────────────────────────────────────────
    else:
        total = df[primary_metric].sum()
        avg   = df[primary_metric].mean()
        std   = df[primary_metric].std()
        max_v = df[primary_metric].max()
        min_v = df[primary_metric].min()
        p25   = df[primary_metric].quantile(0.25)
        p75   = df[primary_metric].quantile(0.75)

        insights.append(f"**Metric analyzed:** `{primary_metric}` across {len(df):,} records")
        insights.append(f"Total: **{total:,.2f}** | Average: **{avg:,.2f}**")
        insights.append(f"Range: {min_v:,.2f} → {max_v:,.2f} | Median IQR: {p25:,.2f} – {p75:,.2f}")
        insights.append(f"Std deviation: **{std:,.2f}**")

        cv = (std / avg * 100) if avg != 0 else 0
        if cv > 50:
            insights.append(f"High variability detected (CV={cv:.1f}%) — significant spread across records.")
        else:
            insights.append(f"Distribution is relatively stable (CV={cv:.1f}%).")

        # Bonus: breakdown by best cat col if available
        if cat_cols:
            group_col = cat_cols[0]
            top_group = df.groupby(group_col)[primary_metric].mean().sort_values(ascending=False).head(3)
            insights.append(f"Top 3 **{group_col}** by avg {primary_metric}:")
            for name, val in top_group.items():
                insights.append(f"• **{name}** → {val:,.2f}")

    # ── Semantic Confidence ───────────────────────────────────────────────────
    top_score = round(results[0]["score"] * 100, 1)
    avg_score = round(sum(r["score"] for r in results) / len(results) * 100, 1)
    insights.append(f"Semantic confidence: **{top_score}%** (top match) | **{avg_score}%** (avg of {len(results)} results)")

    return insights


# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.5rem 0 1rem'>
      <span style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                   background:linear-gradient(135deg,#00e5ff,#7b61ff);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        OTT PULSE
      </span>
      <div style='font-family:DM Mono,monospace;font-size:.65rem;color:#5a7a9a;margin-top:.2rem'>
        ANALYTICS · RAG · 2016–2026
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Search Settings</div>', unsafe_allow_html=True)
    k_results      = st.slider("Results to retrieve", 3, 20, 8)
    show_synthesis = st.toggle("Show Smart Summary", value=True)
    show_raw       = st.toggle("Show raw JSON",      value=False)

    st.markdown('<div class="section-title">Quick Queries</div>', unsafe_allow_html=True)
    quick_queries = [
        "highest revenue platforms",
        "churn rate trends",
        "user growth by year",
        "top subscription plans",
        "regional breakdown",
        "average watch time",
        "content genre popularity",
        "mobile vs desktop usage",
    ]
    selected_quick = None
    sb_cols = st.columns(2)
    for i, q in enumerate(quick_queries):
        if sb_cols[i % 2].button(q, key=f"qq_{i}", use_container_width=True):
            selected_quick = q

    stats = load_stats()
    st.markdown('<div class="section-title">Dataset Info</div>', unsafe_allow_html=True)
    if stats:
        st.markdown(f"**{stats['total_rows']:,}** records · **{len(stats['columns'])}** columns")
        with st.expander("View columns"):
            for col in stats["columns"]:
                st.markdown(f"<span class='chip'>{col}</span>", unsafe_allow_html=True)
    else:
        st.caption("Run `vector.py` to build the index first.")

    st.markdown('<div class="section-title">Setup</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:DM Mono,monospace;font-size:.72rem;color:#5a7a9a;line-height:2.2'>
    1. Place CSV in project root<br>
    2. <code style='color:#00e5ff'>python vector.py</code><br>
    3. <code style='color:#00e5ff'>streamlit run main.py</code>
    </div>""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ott-hero">
  <h1>OTT Analytics RAG</h1>
  <p>semantic search · smart synthesis · 2016 – 2026</p>
</div>""", unsafe_allow_html=True)

# KPI strip
if stats and stats.get("numeric_summary"):
    kpis     = list(stats["numeric_summary"].items())[:5]
    kpi_cols = st.columns(len(kpis) + 1)
    kpi_cols[0].markdown(f"""
    <div class="stat-card">
      <div class="val">{stats['total_rows']:,}</div>
      <div class="lbl">Total Records</div>
    </div>""", unsafe_allow_html=True)
    for i, (cn, info) in enumerate(kpis):
        kpi_cols[i + 1].markdown(f"""
        <div class="stat-card">
          <div class="val">{info['mean']:,.1f}</div>
          <div class="lbl">Avg {cn[:14]}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Search bar ───────────────────────────────────────────────────────────────────
col_s, col_b = st.columns([5, 1])
with col_s:
    query_input = st.text_input(
        "Query", value=selected_quick or "",
        placeholder="e.g. Which platforms had highest churn in 2023?",
        label_visibility="collapsed", key="main_query",
    )
with col_b:
    search_btn = st.button("Search ⟶", use_container_width=True)

st.markdown("""
<div style='margin:.4rem 0 1rem'>
  <span style='font-family:DM Mono,monospace;font-size:.65rem;color:#5a7a9a;margin-right:.5rem'>TRY:</span>
  <span class='chip'>revenue by year</span><span class='chip'>churn analysis</span>
  <span class='chip'>mobile users</span><span class='chip'>top genres</span>
  <span class='chip'>subscription trends</span>
</div>""", unsafe_allow_html=True)

# ── Execute ──────────────────────────────────────────────────────────────────────
final_query = query_input.strip()

if (search_btn or selected_quick) and final_query:

    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        st.error("⚠️ Vector index not found. Run `python vector.py` first.")
        st.stop()

    with st.spinner("Searching vector store…"):
        results = search(final_query, k=k_results)

    if not results:
        st.warning("No results found. Try rephrasing your query.")
        st.stop()

    # Smart Summary
    if show_synthesis:
        st.markdown('<div class="section-title">Smart Summary</div>', unsafe_allow_html=True)
        insights = synthesize(final_query, results)
        rows_html = "".join(
            f'<div class="insight-row"><span class="insight-bullet">›</span>'
            f'<span>{ins}</span></div>'
            for ins in insights
        )
        st.markdown(f"""
        <div class="synthesis-box">
          <span class="label">DATASET SYNTHESIS · FULL DATASET ANALYZED</span>
          {rows_html}
        </div>""", unsafe_allow_html=True)

    # Results tabs
    st.markdown(f'<div class="section-title">Top {len(results)} Semantic Matches</div>',
                unsafe_allow_html=True)
    tab_cards, tab_table, tab_charts = st.tabs(["📇 Cards", "📊 Table", "📈 Charts"])

    with tab_cards:
        for i, res in enumerate(results):
            score_pct = round(res["score"] * 100, 1)
            fields    = res["text"].split(" | ")
            formatted = "<br>".join(
                f"<b style='color:#8ab4d4'>{f.split(':')[0].strip()}:</b>"
                f" {':'.join(f.split(':')[1:]).strip()}" if ":" in f else f
                for f in fields[:8]
            )
            st.markdown(f"""
            <div class="result-card">
              <span class="rank">#{i+1}</span>
              <span class="score-tag">match {score_pct}%</span>
              <div style='margin-top:.5rem'>{formatted}</div>
            </div>""", unsafe_allow_html=True)
            if show_raw and res["record"]:
                with st.expander(f"Raw JSON #{i+1}"):
                    st.json(res["record"])

    with tab_table:
        if results[0]["record"]:
            df_res = pd.DataFrame([r["record"] for r in results])
            df_res.insert(0, "match_%", [round(r["score"] * 100, 1) for r in results])
            st.dataframe(df_res, use_container_width=True, hide_index=True)
        else:
            for i, r in enumerate(results):
                st.text(f"[{i+1}] {r['text']}")

    with tab_charts:
        if results[0]["record"]:
            df_r     = pd.DataFrame([r["record"] for r in results])
            num_cols = df_r.select_dtypes(include="number").columns.tolist()
            cat_cols = df_r.select_dtypes(exclude="number").columns.tolist()
            c1, c2   = st.columns(2)
            if num_cols:
                cn = c1.selectbox("Numeric field", num_cols, key="cn")
                c1.markdown(f"<div class='section-title'>{cn}</div>", unsafe_allow_html=True)
                c1.bar_chart(df_r[cn].reset_index(drop=True))
            if cat_cols:
                cc  = c2.selectbox("Category field", cat_cols, key="cc")
                dst = df_r[cc].value_counts().reset_index()
                dst.columns = [cc, "count"]
                c2.markdown(f"<div class='section-title'>{cc} breakdown</div>", unsafe_allow_html=True)
                c2.bar_chart(dst.set_index(cc))
        else:
            st.info("Re-run vector.py to enable record-level charts.")

else:
    st.markdown("""
    <div style='text-align:center;padding:3rem 2rem;color:#5a7a9a'>
      <div style='font-size:3rem;margin-bottom:1rem'>📡</div>
      <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:600;color:#8ab4d4'>
        Ask anything about your OTT dataset
      </div>
      <div style='font-family:DM Mono,monospace;font-size:.78rem;margin-top:.5rem'>
        Semantic vector search · smart local synthesis · no API key needed
      </div>
    </div>""", unsafe_allow_html=True)
    df = load_df()
    if df is not None:
        st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
