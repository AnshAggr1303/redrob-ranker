"""
app.py — Interactive sandbox demo
Redrob Intelligent Candidate Ranking System — Team 2bits

Upload a sample JSON file (≤100 candidates) and get a ranked CSV back instantly.
Deploy: HuggingFace Spaces (Streamlit SDK)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Redrob Candidate Ranker — 2bits",
    page_icon="🎯",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants — identical to rank.py
# ---------------------------------------------------------------------------
EVALUATION_ANCHOR_DATE = pd.Timestamp("2026-06-13")

IRRELEVANT_TITLES = {
    "accountant", "civil engineer", "hr manager", "human resources manager",
    "mechanical engineer", "graphic designer", "marketing manager",
    "customer support", "customer support executive", "operations manager",
    "business analyst", "content writer", "ui designer", "ux designer",
    "financial analyst", "sales manager", "recruiter",
}

PREFERRED_LOCATIONS = {"noida", "pune", "delhi", "new delhi", "gurgaon", "gurugram", "ncr", "delhi ncr"}
ACCEPTABLE_LOCATIONS = {"hyderabad", "mumbai", "bangalore", "bengaluru", "chennai", "kolkata", "ahmedabad"}

RELEVANT_ASSESSMENTS = {
    "nlp", "python", "machine learning", "information retrieval",
    "embeddings", "recommendation systems", "search", "deep learning",
    "vector search", "ranking", "retrieval", "transformers", "fine-tuning llms", "llm",
}

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "techmahindra",
}

# ---------------------------------------------------------------------------
# Load model and JD embedding
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data(show_spinner="Loading JD embedding...")
def load_jd_embedding():
    jd_path = Path("artefacts/jd_embedding.npy")
    if jd_path.exists():
        return np.load(str(jd_path))
    return None

# ---------------------------------------------------------------------------
# Scoring & Logic Helpers
# ---------------------------------------------------------------------------
def detect_honeypot(candidate: dict) -> bool:
    """Evaluates honeypots on the fly for any uploaded JSON."""
    skills = candidate.get("skills", [])
    
    # Rule 1: Expert proficiency with 0 duration (3+ instances)
    expert_zero = [
        sk["name"] for sk in skills
        if sk.get("proficiency") == "expert" and sk.get("duration_months", 0) == 0
    ]
    if len(expert_zero) >= 3:
        return True

    # Rule 2: Temporal anomaly — claimed duration exceeds calendar span by 12+ months
    for job in candidate.get("career_history", []):
        claimed_months = job.get("duration_months", 0)
        start_str = job.get("start_date")
        if not start_str or claimed_months == 0:
            continue
        try:
            start_date = pd.Timestamp(start_str)
            if job.get("is_current") or not job.get("end_date"):
                end_date = EVALUATION_ANCHOR_DATE
            else:
                end_date = pd.Timestamp(job["end_date"])
            
            physical_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            if claimed_months > physical_months + 12:
                return True
        except ValueError:
            pass

    return False

def normalize_company(name: str) -> str:
    import re
    name = name.lower()
    name = re.sub(r"\b(ltd\.?|limited|technologies|tech|solutions|services|india|pvt\.?|private|inc\.?|corp\.?|llc)\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()

def is_consulting(company_name: str) -> bool:
    norm = normalize_company(company_name)
    return any(cf in norm for cf in CONSULTING_FIRMS)

def location_score(location: str, country: str, willing_to_relocate: bool) -> float:
    loc_lower = (location or "").lower()
    if any(c in loc_lower for c in PREFERRED_LOCATIONS): return 0.08
    if any(c in loc_lower for c in ACCEPTABLE_LOCATIONS): return 0.06
    if country == "India": return 0.04 if willing_to_relocate else 0.02
    return 0.02

def extract_metadata_single(c: dict) -> dict:
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    education = c.get("education", [])
    signals = c.get("redrob_signals", {})

    career_companies = [r.get("company", "") for r in career]
    career_industries = list({r.get("industry", "") for r in career if r.get("industry")})
    all_consulting = len(career_companies) > 0 and all(is_consulting(co) for co in career_companies)
    has_mixed = any(is_consulting(co) for co in career_companies) and not all_consulting

    tier_order = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 5}
    edu_tiers = [e.get("tier", "unknown") for e in education]
    best_tier = min(edu_tiers, key=lambda t: tier_order.get(t, 5)) if edu_tiers else "unknown"

    assess_scores = signals.get("skill_assessment_scores", {}) or {}
    relevant_assess = max((v for k, v in assess_scores.items() if k.lower() in RELEVANT_ASSESSMENTS), default=0.0)

    return {
        "candidate_id": c["candidate_id"],
        "is_honeypot": detect_honeypot(c),
        "years_of_experience": profile.get("years_of_experience", 0) or 0,
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "current_industry": profile.get("current_industry", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "willing_to_relocate": signals.get("willing_to_relocate", False),
        "career_industries_json": json.dumps(career_industries),
        "all_consulting": all_consulting,
        "has_mixed_consulting": has_mixed,
        "best_education_tier": best_tier,
        "relevant_assess_score": relevant_assess,
        "last_active_date": signals.get("last_active_date", "2020-01-01"),
        "open_to_work_flag": signals.get("open_to_work_flag", False),
        "notice_period_days": signals.get("notice_period_days", 90),
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0.5),
        "interview_completion_rate": signals.get("interview_completion_rate", 0.5),
        "github_activity_score": signals.get("github_activity_score", -1),
    }

def build_embedding_text(c: dict) -> str:
    profile = c.get("profile", {})
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    for role in c.get("career_history", []):
        desc = role.get("description", "")
        if desc: parts.append(desc)
    return " ".join(p.strip() for p in parts if p.strip())

def compute_structured_score(row: pd.Series) -> float:
    score = 0.0
    yoe = row["years_of_experience"]
    if 5 <= yoe <= 9: score += 0.10
    elif 4 <= yoe < 5 or 9 < yoe <= 11: score += 0.05
    elif yoe > 12: score -= 0.05
    
    score += location_score(row["location"], row["country"], bool(row["willing_to_relocate"]))
    
    notice = row["notice_period_days"]
    if notice <= 30: score += 0.06
    elif notice <= 60: score += 0.02
    
    if row["best_education_tier"] == "tier_1": score += 0.03
    elif row["best_education_tier"] == "tier_2": score += 0.01
    
    if row["has_mixed_consulting"]: score -= 0.03
    
    # Skill assessment logic is handled completely inside S_struct here
    if row["relevant_assess_score"] >= 70: score += 0.02
    return score

def compute_availability_multiplier(row: pd.Series) -> float:
    days_since = (EVALUATION_ANCHOR_DATE - pd.Timestamp(row["last_active_date"])).days
    f_active = 1.00 if days_since <= 30 else 0.85 if days_since <= 90 else 0.65 if days_since <= 180 else 0.40
    f_open = 1.00 if row["open_to_work_flag"] else 0.85
    f_response = 0.60 + (float(row["recruiter_response_rate"]) * 0.40)
    icr = float(row["interview_completion_rate"])
    f_interview = 1.00 if icr >= 0.80 else 0.90 if icr >= 0.50 else 0.75
    return float(np.clip(f_active * f_open * f_response * f_interview, 0.35, 1.0))

def extract_primary_strength(row: pd.Series) -> str:
    yoe = row["years_of_experience"]
    github = row["github_activity_score"]
    title = row["current_title"]
    is_relevant = any(t in title.lower() for t in {"machine learning", "ml engineer", "ai engineer", "nlp", "search engineer", "ranking", "retrieval", "applied scientist", "research engineer"})
    
    if is_relevant and 5 <= yoe <= 9: return f"{yoe} years in applied ML/AI engineering with a relevant title ({title})"
    if github >= 60: return f"strong open-source activity (GitHub score: {int(github)}/100)"
    if row["relevant_assess_score"] >= 70: return f"a verified platform assessment score of {int(row['relevant_assess_score'])}/100 on a relevant AI/ML skill"
    return f"engineering experience at {row['current_company']} in the {row['current_industry'] or 'technology'} sector"

def extract_primary_concern(row: pd.Series):
    notice = row["notice_period_days"]
    days_since = (EVALUATION_ANCHOR_DATE - pd.Timestamp(row["last_active_date"])).days
    if notice > 90: return f"a long notice period of {int(notice)} days"
    if days_since > 180: return f"profile inactivity ({days_since} days since last login)"
    if row["has_mixed_consulting"]: return "partial consulting-firm tenure in career history"
    if row["country"] != "India" and row["willing_to_relocate"]: return f"an overseas location ({row['country']}) requiring relocation"
    return None

def generate_reasoning(row: pd.Series, rank: int) -> str:
    strength = extract_primary_strength(row)
    concern = extract_primary_concern(row)
    if rank <= 25:
        return f"Exceptional fit driven by {strength}." + (f" Note: {concern}." if concern else "")
    elif rank <= 60:
        return f"Solid technical baseline with {strength}, though limited by {concern or 'limited platform engagement signals'}."
    return f"Ranked primarily due to {concern or 'overall profile gaps'}. However, {strength} justifies inclusion in the extended pipeline."

# ---------------------------------------------------------------------------
# Main Ranking Function
# ---------------------------------------------------------------------------
def rank_candidates(candidates: list, model, jd_vec: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame([extract_metadata_single(c) for c in candidates])

    # Dynamic Embedding
    texts = [build_embedding_text(c) for c in candidates]
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / np.clip(norms, 1e-9, None)
    jd_normed = jd_vec / np.linalg.norm(jd_vec)
    df["S_sem"] = np.clip(normed @ jd_normed, 0.0, 1.0)

    # Disqualifiers
    mask_honeypot   = df["is_honeypot"]
    mask_consulting = df["all_consulting"].astype(bool)
    mask_irrelevant = df["current_title"].str.lower().str.strip().isin(IRRELEVANT_TITLES)
    mask_foreign    = (df["country"] != "India") & (~df["willing_to_relocate"].astype(bool))
    disqualified    = mask_honeypot | mask_consulting | mask_irrelevant | mask_foreign

    # Math
    df["S_struct"]   = df.apply(compute_structured_score, axis=1)
    df["avail_mult"] = df.apply(compute_availability_multiplier, axis=1)
    df["github_norm"] = (df["github_activity_score"].clip(lower=0) / 100.0).clip(0, 1)

    # Final formula perfectly matches rank.py structure
    df["S_final"] = (
        (df["S_sem"] * 0.55 + df["S_struct"] * 0.30) * df["avail_mult"] 
        + df["github_norm"] * 0.05
    )
    df.loc[disqualified, "S_final"] = 0.0

    # Sort & Reason
    top_n = min(100, (~disqualified).sum())
    ranked = df[~disqualified].sort_values(["S_final", "candidate_id"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    ranked["reasoning"] = ranked.apply(lambda r: generate_reasoning(r, int(r["rank"])), axis=1)

    return ranked, df[disqualified], disqualified.sum()

# ===========================================================================
# UI — Custom CSS
# ===========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ── Root tokens ─────────────────────────────────────────────────── */
:root {
    --indigo:     #4F46E5;
    --indigo-lt:  #EEF2FF;
    --teal:       #0D9488;
    --amber:      #D97706;
    --rose:       #E11D48;
    --green:      #059669;
    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  14px;
    --radius-xl:  20px;
}

/* ── Base ────────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif !important; }
#MainMenu, footer, header   { visibility: hidden; }
.block-container            { padding-top: 1.5rem !important; max-width: 1200px !important; }

/* ── Hero ────────────────────────────────────────────────────────── */
.rr-hero {
    background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #1E3A5F 100%);
    border-radius: var(--radius-xl);
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.rr-hero::before {
    content: ''; position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px; border-radius: 50%;
    background: rgba(79,70,229,0.25);
}
.rr-hero::after {
    content: ''; position: absolute;
    bottom: -60px; left: 30%;
    width: 280px; height: 280px; border-radius: 50%;
    background: rgba(13,148,136,0.15);
}
.rr-hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 11px; font-weight: 500; letter-spacing: 0.06em;
    color: rgba(255,255,255,0.8); text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.rr-hero-dot {
    display: inline-block;
    width: 7px; height: 7px; border-radius: 50%;
    background: #34D399;
    box-shadow: 0 0 8px #34D399;
    animation: rr-pulse 2s infinite;
}
@keyframes rr-pulse { 0%,100% { opacity:1; } 50% { opacity:0.35; } }
.rr-hero h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2rem !important; font-weight: 700 !important;
    color: #FFFFFF !important;
    margin: 0 0 0.4rem !important; line-height: 1.2 !important;
}
.rr-hero p {
    font-size: 0.95rem; color: rgba(255,255,255,0.65);
    margin: 0; max-width: 520px;
}

/* ── Upload zone ─────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #CBD5E1 !important;
    border-radius: var(--radius-lg) !important;
    background: #F8FAFC !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #4F46E5 !important; }

/* ── Primary button ──────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #4F46E5 !important;
    color: #fff !important; border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 500 !important; font-size: 0.9rem !important;
    padding: 0.6rem 1.6rem !important;
    transition: opacity 0.15s, transform 0.1s !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover  { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

/* ── Metric cards ────────────────────────────────────────────────── */
.rr-metric-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin: 1.5rem 0;
}
.rr-metric {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: var(--radius-lg);
    padding: 1rem 1.25rem;
    position: relative; overflow: hidden;
}
.rr-metric-icon {
    width: 32px; height: 32px; border-radius: var(--radius-sm);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; margin-bottom: 0.75rem;
}
.rr-metric-label {
    font-size: 11px; font-weight: 500; letter-spacing: 0.05em;
    color: #94A3B8; text-transform: uppercase; margin-bottom: 4px;
}
.rr-metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem; font-weight: 600;
    color: #0F172A; line-height: 1;
}
.rr-metric-sub { font-size: 11px; color: #94A3B8; margin-top: 4px; }
.rr-metric-bar { position: absolute; bottom: 0; left: 0; right: 0; height: 3px; }

/* ── Section header ──────────────────────────────────────────────── */
.rr-section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 2rem 0 1rem; padding-bottom: 0.75rem;
    border-bottom: 1px solid #E2E8F0;
}
.rr-section-header h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important; font-weight: 600 !important;
    color: #0F172A !important; margin: 0 !important;
}
.rr-section-icon {
    width: 28px; height: 28px; background: #EEF2FF;
    border-radius: var(--radius-sm);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
}

/* ── Success banner ──────────────────────────────────────────────── */
.rr-success {
    display: flex; align-items: center; gap: 10px;
    background: #ECFDF5; border: 1px solid #A7F3D0;
    border-radius: var(--radius-md);
    padding: 0.75rem 1rem;
    font-size: 0.875rem; color: #065F46; margin: 1rem 0;
}

/* ── Download button ─────────────────────────────────────────────── */
.stDownloadButton > button {
    background: #0D9488 !important; color: #fff !important;
    border: none !important; border-radius: var(--radius-md) !important;
    font-weight: 500 !important; width: 100% !important;
}
.stDownloadButton > button:hover { opacity: 0.88 !important; }

/* ── Disqualified badge ──────────────────────────────────────────── */
.rr-disq-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: #FFF1F2; border: 1px solid #FECDD3;
    border-radius: 999px; padding: 3px 10px;
    font-size: 11px; font-weight: 500; color: #9F1239;
}

/* ── Empty state ─────────────────────────────────────────────────── */
.rr-empty {
    text-align: center; padding: 3rem 2rem;
    border: 1.5px dashed #CBD5E1; border-radius: 14px;
    background: #F8FAFC; margin-top: 1rem;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] { background: #0F172A !important; }
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
.rr-sidebar-title {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: rgba(255,255,255,0.35) !important;
    margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.rr-step { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 1.25rem; }
.rr-step-num {
    min-width: 22px; height: 22px; border-radius: 50%;
    background: rgba(79,70,229,0.35); border: 1px solid rgba(79,70,229,0.55);
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600; color: #A5B4FC !important; margin-top: 1px;
}
.rr-step-title { font-size: 13px; font-weight: 500; color: rgba(255,255,255,0.9) !important; margin: 0 0 2px; }
.rr-step-desc  { font-size: 11.5px; line-height: 1.5; color: rgba(255,255,255,0.42) !important; margin: 0; }
.rr-formula {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: var(--radius-md);
    padding: 0.75rem 1rem;
    font-family: 'Courier New', monospace;
    font-size: 11px; line-height: 1.7;
    color: #A5B4FC !important; margin-top: 1rem;
}
.rr-sidebar-footer {
    font-size: 11px; color: rgba(255,255,255,0.22) !important;
    text-align: center; margin-top: 2rem; padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}

/* ── Expander ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: var(--radius-md) !important;
    background: #F8FAFC !important;
}
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# Hero
# ===========================================================================
st.markdown("""
<div class="rr-hero">
    <div class="rr-hero-badge">
        <span class="rr-hero-dot"></span>
        India Runs Data &amp; AI Challenge 2026
    </div>
    <h1>Redrob Candidate Ranker</h1>
    <p>Five-layer ML pipeline — semantic embeddings, structured signals, and availability scoring — to surface the right engineers from any candidate pool.</p>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# Upload row
# ===========================================================================
col_up, col_ctrl = st.columns([3, 1], gap="large")

with col_up:
    st.markdown("##### Upload candidates file")
    uploaded = st.file_uploader(
        "Drop a JSON file here or click to browse",
        type=["json"],
        label_visibility="collapsed",
        help="JSON array of candidate objects matching the Redrob schema.",
    )

with col_ctrl:
    st.markdown("##### Display top N")
    top_n_display = st.slider("top_n", 5, 100, 20, 5, label_visibility="collapsed")
    st.markdown(
        f"<p style='font-size:12px;color:#94A3B8;margin-top:-8px;'>Showing top <strong>{top_n_display}</strong> of up to 100</p>",
        unsafe_allow_html=True,
    )
    run_btn = st.button("🚀 Run ranking", type="primary", use_container_width=True)

# ===========================================================================
# Parse upload
# ===========================================================================
candidates = None
if uploaded:
    try:
        raw = json.loads(uploaded.read().decode("utf-8"))
        candidates = raw if isinstance(raw, list) else list(raw.values())
        st.markdown(
            f'<div class="rr-success">✅ <strong>{len(candidates)}</strong> candidate profiles loaded from <em>{uploaded.name}</em></div>',
            unsafe_allow_html=True,
        )
        if len(candidates) > 500:
            st.warning("⚠️ Large file detected. For best demo performance, use ≤100 candidates.")
    except Exception as e:
        st.error(f"Could not parse JSON: {e}")

# ===========================================================================
# Run ranking
# ===========================================================================
if run_btn and candidates:
    with st.spinner("Loading embedding model and JD vector…"):
        model  = load_model()
        jd_vec = load_jd_embedding()

    if jd_vec is None:
        st.error("JD embedding not found — ensure `artefacts/jd_embedding.npy` exists.")
        st.stop()

    progress = st.progress(0, text="Encoding candidate profiles…")
    t0 = time.time()
    ranked, disq_df, n_disq = rank_candidates(candidates, model, jd_vec)
    elapsed = time.time() - t0
    progress.progress(100, text="Done!")
    time.sleep(0.35)
    progress.empty()

    # ── Metric cards ──────────────────────────────────────────────────────
    pct_eligible = round(len(ranked) / len(candidates) * 100) if candidates else 0
    avg_score    = round(float(ranked["S_final"].mean()), 3) if len(ranked) else 0
    top_score    = round(float(ranked["S_final"].iloc[0]), 3) if len(ranked) else 0

    st.markdown(f"""
    <div class="rr-metric-grid">
        <div class="rr-metric">
            <div class="rr-metric-icon" style="background:#EEF2FF;">📥</div>
            <div class="rr-metric-label">Total uploaded</div>
            <div class="rr-metric-value">{len(candidates)}</div>
            <div class="rr-metric-sub">candidates in file</div>
            <div class="rr-metric-bar" style="background:#4F46E5;"></div>
        </div>
        <div class="rr-metric">
            <div class="rr-metric-icon" style="background:#F0FDFA;">✅</div>
            <div class="rr-metric-label">Eligible</div>
            <div class="rr-metric-value">{len(ranked)}</div>
            <div class="rr-metric-sub">{pct_eligible}% passed all filters</div>
            <div class="rr-metric-bar" style="background:#0D9488;"></div>
        </div>
        <div class="rr-metric">
            <div class="rr-metric-icon" style="background:#FFF1F2;">🚫</div>
            <div class="rr-metric-label">Disqualified</div>
            <div class="rr-metric-value">{n_disq}</div>
            <div class="rr-metric-sub">honeypot · consulting · irrelevant</div>
            <div class="rr-metric-bar" style="background:#E11D48;"></div>
        </div>
        <div class="rr-metric">
            <div class="rr-metric-icon" style="background:#FFFBEB;">⚡</div>
            <div class="rr-metric-label">Runtime</div>
            <div class="rr-metric-value">{elapsed:.2f}s</div>
            <div class="rr-metric-sub">avg {avg_score} · top {top_score}</div>
            <div class="rr-metric-bar" style="background:#D97706;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Results table ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="rr-section-header">
        <div class="rr-section-icon">🏆</div>
        <h3>Ranked candidates</h3>
    </div>
    """, unsafe_allow_html=True)

    display = ranked[["rank", "candidate_id", "S_final", "current_title",
                       "years_of_experience", "location", "notice_period_days", "reasoning"]
                     ].head(top_n_display).copy()
    display.columns = ["Rank", "Candidate ID", "Score", "Title", "YOE", "Location", "Notice (days)", "Reasoning"]
    display["Score"] = display["Score"].round(4)
    display["YOE"]   = display["YOE"].astype(int)

    # Relative percentile colouring — always meaningful regardless of score range
    scores = display["Score"]
    p66 = float(scores.quantile(0.66))
    p33 = float(scores.quantile(0.33))

    def _colour_score(val):
        if val >= p66:   return "background-color:#DCFCE7;color:#166534;font-weight:500;"
        elif val >= p33: return "background-color:#FEF9C3;color:#854D0E;font-weight:500;"
        else:            return "background-color:#FEE2E2;color:#991B1B;font-weight:500;"

    styled = display.style.map(_colour_score, subset=["Score"])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank":         st.column_config.NumberColumn(width="small"),
            "Score":        st.column_config.NumberColumn(format="%.4f", width="small"),
            "YOE":          st.column_config.NumberColumn(format="%d yrs", width="small"),
            "Notice (days)":st.column_config.NumberColumn(format="%d d", width="small"),
            "Reasoning":    st.column_config.TextColumn(width="large"),
        },
    )

    # ── Score distribution chart ───────────────────────────────────────────
    st.markdown("""
    <div class="rr-section-header" style="margin-top:2rem;">
        <div class="rr-section-icon">📊</div>
        <h3>Score distribution</h3>
    </div>
    """, unsafe_allow_html=True)

    try:
        import altair as alt
        chart_df = ranked[["rank", "S_final"]].rename(columns={"rank": "Rank", "S_final": "Score"})
        chart_df["Band"] = pd.cut(
            chart_df["Score"],
            bins=[0, float(chart_df["Score"].quantile(0.33)),
                     float(chart_df["Score"].quantile(0.66)), 1.0],
            labels=["Lower third", "Middle third", "Top third"],
        ).astype(str)

        area = (
            alt.Chart(chart_df)
            .mark_area(
                interpolate="monotone",
                line={"color": "#4F46E5", "strokeWidth": 2},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="#4F46E5", offset=0),
                        alt.GradientStop(color="#EEF2FF", offset=1),
                    ],
                    x1=1, x2=1, y1=1, y2=0,
                ),
                opacity=0.85,
            )
            .encode(
                x=alt.X("Rank:Q", axis=alt.Axis(title="Rank", grid=False)),
                y=alt.Y("Score:Q",
                        axis=alt.Axis(title="Final score", format=".3f", grid=True),
                        scale=alt.Scale(domain=[0, float(chart_df["Score"].max()) * 1.15])),
                tooltip=["Rank:Q", alt.Tooltip("Score:Q", format=".4f"), "Band:N"],
            )
            .properties(height=260)
            .configure_view(strokeWidth=0)
            .configure_axis(labelFont="Inter", titleFont="Inter", labelFontSize=11, titleFontSize=12)
        )
        st.altair_chart(area, use_container_width=True)
    except ImportError:
        chart_df2 = pd.DataFrame({"Score": ranked["S_final"].values}, index=ranked["rank"].values)
        chart_df2.index.name = "Rank"
        st.line_chart(chart_df2)

    # ── Export + disqualified ──────────────────────────────────────────────
    dl_col, disq_col = st.columns([1, 1], gap="large")

    with dl_col:
        st.markdown("""
        <div class="rr-section-header">
            <div class="rr-section-icon">⬇️</div>
            <h3>Export results</h3>
        </div>
        """, unsafe_allow_html=True)
        output_csv = ranked[["candidate_id", "rank", "S_final", "reasoning"]].copy()
        output_csv.columns = ["candidate_id", "rank", "score", "reasoning"]
        output_csv["score"] = output_csv["score"].round(6)
        st.download_button(
            label="⬇️  Download team_2bits.csv",
            data=output_csv.to_csv(index=False).encode("utf-8"),
            file_name="team_2bits.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown(
            f"<p style='font-size:12px;color:#94A3B8;margin-top:6px;'>CSV — {len(ranked)} rows · candidate_id, rank, score, reasoning</p>",
            unsafe_allow_html=True,
        )

    with disq_col:
        if n_disq > 0:
            st.markdown(f"""
            <div class="rr-section-header">
                <div class="rr-section-icon">🚫</div>
                <h3>Disqualified <span class="rr-disq-badge">{n_disq} profiles</span></h3>
            </div>
            <p style="font-size:12px;color:#94A3B8;">Expand below to inspect removed profiles.</p>
            """, unsafe_allow_html=True)

    # Full-width expander so it renders reliably
    if n_disq > 0:
        with st.expander(f"🚫 View {n_disq} disqualified candidates", expanded=False):
            disq_display = disq_df[["candidate_id", "current_title", "country",
                                    "is_honeypot", "all_consulting", "willing_to_relocate"]
                                   ].reset_index(drop=True)
            st.dataframe(
                disq_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "candidate_id":    st.column_config.TextColumn("Candidate ID"),
                    "current_title":   st.column_config.TextColumn("Title"),
                    "country":         st.column_config.TextColumn("Country"),
                    "is_honeypot":     st.column_config.CheckboxColumn("Honeypot?"),
                    "all_consulting":  st.column_config.CheckboxColumn("All consulting?"),
                    "willing_to_relocate": st.column_config.CheckboxColumn("Willing to relocate?"),
                },
            )

elif run_btn and not candidates:
    st.warning("Upload a candidates file first, then click Run ranking.")

elif not uploaded:
    st.markdown("""
    <div class="rr-empty">
        <div style="font-size:2.5rem;margin-bottom:0.75rem;">📂</div>
        <p style="font-size:0.95rem;color:#475569;margin:0 0 0.4rem;font-weight:500;">No file uploaded yet</p>
        <p style="font-size:0.825rem;color:#94A3B8;margin:0;">Use <code>data/sample_candidates.json</code> from the repo to get started</p>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================================
# Sidebar — pipeline explainer
# ===========================================================================
with st.sidebar:
    st.markdown('<div class="rr-sidebar-title">Pipeline layers</div>', unsafe_allow_html=True)

    steps = [
        ("Hard filters",
         "Removes honeypots (on-the-fly detection), consulting-only careers, irrelevant titles, and non-India profiles unwilling to relocate."),
        ("Semantic score · 55%",
         "Cosine similarity between candidate narrative (headline + summary + career) and the JD signal vector via all-MiniLM-L6-v2."),
        ("Structured score · 30%",
         "YOE band (5–9 yrs ideal), location tier, notice period, education tier, consulting penalty, assessment score."),
        ("Availability multiplier",
         "Scales the combined score by last-active date, open-to-work flag, recruiter response rate, and interview completion rate."),
        ("GitHub signal · 5%",
         "Normalised GitHub activity score added on top of the multiplied base."),
    ]

    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="rr-step">
            <div class="rr-step-num">{i}</div>
            <div>
                <p class="rr-step-title">{title}</p>
                <p class="rr-step-desc">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rr-formula">
S_final =<br>
&nbsp;&nbsp;( S_sem &times; 0.55<br>
&nbsp;&nbsp;+ S_struct &times; 0.30 )<br>
&nbsp;&nbsp;&times; avail_mult<br>
&nbsp;&nbsp;+ github_norm &times; 0.05
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="rr-sidebar-footer">Team 2bits · Redrob India Runs Challenge 2026</div>',
        unsafe_allow_html=True,
    )