"""
rank.py — Phase 2 runtime ranker
Redrob Intelligent Candidate Ranking System

Loads pre-computed artefacts and ranks 100,000 candidates in <5 minutes
on a single CPU. No network access required.

Usage:
    python src/rank.py \
        --candidates ./data/candidates.jsonl \
        --out ./outputs/submission.csv

Artefacts required in ./artefacts/:
    embeddings.npy, jd_embedding.npy, metadata.parquet, honeypot_ids.txt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

# Fixed anchor date — guarantees deterministic availability decay
# across all judge evaluation environments regardless of when they run this.
EVALUATION_ANCHOR_DATE = pd.Timestamp("2026-06-13")

ARTEFACTS_DIR = Path("./artefacts")

# ---------------------------------------------------------------------------
# Hard disqualifier — irrelevant titles
# Whole-word match, case-insensitive
# ---------------------------------------------------------------------------
IRRELEVANT_TITLES = {
    "accountant", "civil engineer", "hr manager", "human resources manager",
    "mechanical engineer", "graphic designer", "marketing manager",
    "customer support", "customer support executive", "operations manager",
    "business analyst", "content writer", "ui designer", "ux designer",
    "financial analyst", "sales manager", "recruiter",
}

# ---------------------------------------------------------------------------
# Location scoring — preferred and acceptable Indian cities
# Location field can be "Chennai, Tamil Nadu" or just "Gurgaon"
# ---------------------------------------------------------------------------
PREFERRED_LOCATIONS = {
    "noida", "pune", "delhi", "new delhi", "gurgaon", "gurugram",
    "ncr", "delhi ncr",
}
ACCEPTABLE_LOCATIONS = {
    "hyderabad", "mumbai", "bangalore", "bengaluru", "chennai",
    "kolkata", "ahmedabad",
}

# ---------------------------------------------------------------------------
# Relevant skill assessments for this JD
# ---------------------------------------------------------------------------
RELEVANT_ASSESSMENTS = {
    "nlp", "python", "machine learning", "information retrieval",
    "embeddings", "recommendation systems", "search", "deep learning",
    "vector search", "ranking", "retrieval", "transformers",
    "fine-tuning llms", "llm",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_artefacts(artefacts_dir: Path):
    """Load all pre-computed artefacts. Target: <3 seconds."""
    t0 = time.time()

    embeddings = np.load(artefacts_dir / "embeddings.npy")       # (100000, 384)
    jd_vec     = np.load(artefacts_dir / "jd_embedding.npy")     # (384,)
    metadata   = pd.read_parquet(artefacts_dir / "metadata.parquet")

    with open(artefacts_dir / "honeypot_ids.txt") as f:
        honeypot_ids = set(line.strip() for line in f if line.strip())

    print(f"      Artefacts loaded in {time.time()-t0:.2f}s")
    print(f"      Embeddings: {embeddings.shape} | Metadata: {metadata.shape}")
    print(f"      Honeypot IDs: {len(honeypot_ids):,}")
    return embeddings, jd_vec, metadata, honeypot_ids


def location_score(location: str, country: str, willing_to_relocate: bool) -> float:
    """Return location bonus based on proximity to JD preferred cities."""
    if country != "India" and not willing_to_relocate:
        return 0.0  # handled as disqualifier in Layer 1, but safe fallback

    loc_lower = location.lower() if location else ""

    # Check preferred cities first
    for city in PREFERRED_LOCATIONS:
        if city in loc_lower:
            return 0.08

    # Check acceptable cities
    for city in ACCEPTABLE_LOCATIONS:
        if city in loc_lower:
            return 0.06

    # India but unlisted city
    if country == "India":
        return 0.04 if willing_to_relocate else 0.02

    # Outside India but willing to relocate
    return 0.02


def compute_structured_score(row: pd.Series) -> float:
    """
    Layer 3 — structured scoring bonus.
    All additive. Max theoretical ~0.40.
    """
    score = 0.0
    yoe = row["years_of_experience"]

    # YOE band
    if 5 <= yoe <= 9:
        score += 0.10
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        score += 0.05
    elif yoe > 12:
        score -= 0.05  # Over-experienced: JD flags 15yr candidates without senior judgment

    # Location
    score += location_score(
        row["location"],
        row["country"],
        bool(row["willing_to_relocate"]),
    )

    # Notice period
    notice = row["notice_period_days"]
    if notice <= 30:
        score += 0.06
    elif notice <= 60:
        score += 0.02

    # GitHub activity
    github = row["github_activity_score"]
    if github >= 70:
        score += 0.06
    elif github >= 40:
        score += 0.04

    # Education tier
    if row["best_education_tier"] == "tier_1":
        score += 0.03
    elif row["best_education_tier"] == "tier_2":
        score += 0.01

    # Consulting partial penalty (mixed history, not full consulting)
    if row["has_mixed_consulting"]:
        score -= 0.03

    # Skill assessment bonus
    if row["relevant_assess_score"] >= 70:
        score += 0.02

    return score


def compute_availability_multiplier(row: pd.Series) -> float:
    """
    Layer 4 — availability multiplier.
    Product of four sub-multipliers, clamped to [0.35, 1.0].
    """
    # f_active — recency of last login
    days_since = (EVALUATION_ANCHOR_DATE - pd.Timestamp(row["last_active_date"])).days
    if days_since <= 30:
        f_active = 1.00
    elif days_since <= 90:
        f_active = 0.85
    elif days_since <= 180:
        f_active = 0.65
    else:
        f_active = 0.40

    # f_open — open to work flag
    f_open = 1.00 if row["open_to_work_flag"] else 0.85

    # f_response — recruiter response rate (linear 0.6 → 1.0)
    response_rate = float(row["recruiter_response_rate"])
    f_response = 0.60 + (response_rate * 0.40)

    # f_interview — interview completion rate
    icr = float(row["interview_completion_rate"])
    if icr >= 0.80:
        f_interview = 1.00
    elif icr >= 0.50:
        f_interview = 0.90
    else:
        f_interview = 0.75

    mult = f_active * f_open * f_response * f_interview
    return float(np.clip(mult, 0.35, 1.0))


# ---------------------------------------------------------------------------
# Reasoning generator
# ---------------------------------------------------------------------------

def extract_primary_strength(row: pd.Series) -> str:
    """Extract the single strongest positive signal for this candidate."""
    industries = json.loads(row["career_industries_json"])
    yoe = row["years_of_experience"]
    github = row["github_activity_score"]
    assess = row["relevant_assess_score"]
    title = row["current_title"]

    # Priority 1: relevant title + YOE in sweet spot
    relevant_titles = {
        "machine learning", "ml engineer", "ai engineer", "nlp", "data scientist",
        "search engineer", "ranking", "retrieval", "recommendation", "applied scientist",
        "research engineer", "platform engineer", "backend engineer", "software engineer",
    }
    title_lower = title.lower()
    is_relevant_title = any(t in title_lower for t in relevant_titles)

    if is_relevant_title and 5 <= yoe <= 9:
        return f"{yoe} years in applied ML/AI engineering with a relevant title ({title})"

    # Priority 2: strong GitHub signal
    if github >= 60:
        return f"strong open-source activity (GitHub score: {int(github)}/100)"

    # Priority 3: verified assessment score
    if assess >= 70:
        return f"a verified platform assessment score of {int(assess)}/100 on a relevant AI/ML skill"

    # Priority 4: relevant title outside YOE sweet spot (still a signal)
    if is_relevant_title and yoe > 0:
        return f"{yoe} years in applied ML/AI engineering with a relevant title ({title})"

    # Priority 5: relevant industry background
    ml_industries = {"Artificial Intelligence", "Machine Learning", "Technology", "SaaS", "FinTech"}
    matching = [i for i in industries if i in ml_industries]
    if matching:
        return f"a background in {matching[0]} that aligns with the role's technical domain"

    # Fallback — use actual company name, never a generic phrase
    company = row["current_company"] or "current employer"
    industry = row["current_industry"] or "technology"
    return f"engineering experience at {company} in the {industry} sector"


def extract_primary_concern(row: pd.Series) -> str | None:
    """
    Extract the single most significant concern as a NOUN PHRASE.
    Must work grammatically after: "due to X", "limited by X", "Note: X".
    No verbs — noun phrases only.
    """
    notice = row["notice_period_days"]
    days_since = (EVALUATION_ANCHOR_DATE - pd.Timestamp(row["last_active_date"])).days
    response_rate = float(row["recruiter_response_rate"])
    yoe = row["years_of_experience"]
    country = row["country"]
    willing = row["willing_to_relocate"]

    # Priority order — worst concern first
    if notice > 90:
        return f"a long notice period of {int(notice)} days"

    if days_since > 180:
        return f"profile inactivity ({days_since} days since last login)"

    if row["has_mixed_consulting"]:
        return "partial consulting-firm tenure in career history"

    if country != "India" and willing:
        return f"an overseas location ({country}) requiring relocation before joining"

    if yoe < 4 or yoe > 12:
        return f"an experience level of {yoe} years outside the preferred 5–9 range"

    if response_rate < 0.30:
        return f"a low recruiter response rate of {int(response_rate * 100)}% on platform"

    if days_since > 90:
        return f"reduced platform activity ({days_since} days since last login)"

    return None


def generate_reasoning(row: pd.Series, rank: int) -> str:
    """
    Generate a data-driven, rank-consistent reasoning string.
    Tone degrades naturally from enthusiastic (top 25) to cautious (bottom 40).
    Every value comes from actual candidate fields — no hallucination.
    All concern strings are noun phrases so "due to X" and "limited by X" are always grammatical.
    """
    strength = extract_primary_strength(row)
    concern = extract_primary_concern(row)

    if rank <= 25:
        base = f"Exceptional fit driven by {strength}."
        if concern:
            base += f" Note: {concern}."
        return base

    elif rank <= 60:
        concern_text = concern or "limited platform engagement signals"
        return (
            f"Solid technical baseline with {strength}, "
            f"though limited by {concern_text}."
        )

    else:
        concern_text = concern or "overall profile gaps relative to JD requirements"
        return (
            f"Ranked primarily due to {concern_text}. "
            f"However, {strength} justifies inclusion in the extended pipeline."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Redrob Phase 2 Runtime Ranker")
    parser.add_argument("--candidates", default="./data/candidates.jsonl",
                        help="Path to candidates.jsonl (used for candidate_id ordering only)")
    parser.add_argument("--artefacts", default="./artefacts",
                        help="Directory containing pre-computed artefacts")
    parser.add_argument("--out", default="./outputs/submission.csv",
                        help="Output path for ranked CSV")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load artefacts
    # ------------------------------------------------------------------
    print("\n[1/5] Loading artefacts ...")
    embeddings, jd_vec, metadata, honeypot_ids = load_artefacts(Path(args.artefacts))

    # ------------------------------------------------------------------
    # Step 2: Layer 1 — hard disqualifiers (vectorized)
    # ------------------------------------------------------------------
    print("\n[2/5] Applying hard disqualifiers ...")
    t0 = time.time()

    # Honeypot
    mask_honeypot = metadata["candidate_id"].isin(honeypot_ids)

    # Consulting-only career
    mask_consulting = metadata["all_consulting"].astype(bool)

    # Irrelevant title — vectorized string check
    def is_irrelevant_title(title: str) -> bool:
        return str(title).lower().strip() in IRRELEVANT_TITLES

    mask_irrelevant = metadata["current_title"].apply(is_irrelevant_title)

    # Outside India and not willing to relocate
    mask_foreign = (metadata["country"] != "India") & (~metadata["willing_to_relocate"].astype(bool))

    # Combined disqualifier mask
    disqualified = mask_honeypot | mask_consulting | mask_irrelevant | mask_foreign

    n_honeypot    = mask_honeypot.sum()
    n_consulting  = mask_consulting.sum()
    n_irrelevant  = mask_irrelevant.sum()
    n_foreign     = mask_foreign.sum()
    n_disqualified = disqualified.sum()

    print(f"      Honeypots:         {n_honeypot:>6,}")
    print(f"      Consulting-only:   {n_consulting:>6,}")
    print(f"      Irrelevant title:  {n_irrelevant:>6,}")
    print(f"      Foreign/no-relo:   {n_foreign:>6,}")
    print(f"      Total disqualified:{n_disqualified:>6,} / {len(metadata):,}")
    print(f"      Eligible pool:     {(~disqualified).sum():>6,}")
    print(f"      Done in {time.time()-t0:.2f}s")

    # ------------------------------------------------------------------
    # Step 3: Layer 2 — semantic score (vectorized cosine similarity)
    # ------------------------------------------------------------------
    print("\n[3/5] Computing semantic scores ...")
    t0 = time.time()

    # L2-normalize all embeddings and JD vector
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-9, None)
    jd_normed = jd_vec / np.linalg.norm(jd_vec)

    # Cosine similarity via dot product — shape (100000,)
    S_sem = (normed @ jd_normed).astype(np.float64)
    S_sem = np.clip(S_sem, 0.0, 1.0)

    metadata = metadata.copy()
    metadata["S_sem"] = S_sem

    print(f"      Semantic scores — min: {S_sem.min():.4f} | "
          f"mean: {S_sem.mean():.4f} | max: {S_sem.max():.4f}")
    print(f"      Done in {time.time()-t0:.2f}s")

    # ------------------------------------------------------------------
    # Step 4: Layers 3 & 4 — structured score + availability multiplier
    # ------------------------------------------------------------------
    print("\n[4/5] Computing structured scores and availability multipliers ...")
    t0 = time.time()

    metadata["S_struct"]   = metadata.apply(compute_structured_score, axis=1)
    metadata["avail_mult"] = metadata.apply(compute_availability_multiplier, axis=1)

    # GitHub normalized bonus [0, 0.04]
    github_raw = metadata["github_activity_score"].clip(lower=0)
    metadata["github_norm"] = (github_raw / 100.0).clip(0, 1)

    # Assessment bonus
    metadata["assess_bonus"] = (metadata["relevant_assess_score"] >= 70).astype(float) * 0.02

    # Layer 5 — final formula
    metadata["S_final"] = (
        (metadata["S_sem"] * 0.55 + metadata["S_struct"] * 0.30)
        * metadata["avail_mult"]
        + metadata["github_norm"] * 0.05
        + metadata["assess_bonus"] * 0.05
    )

    # Zero out disqualified candidates
    metadata.loc[disqualified, "S_final"] = 0.0

    print(f"      Structured scores — mean: {metadata['S_struct'].mean():.4f}")
    print(f"      Availability mult — mean: {metadata['avail_mult'].mean():.4f}")
    print(f"      Final scores (eligible) — "
          f"mean: {metadata.loc[~disqualified, 'S_final'].mean():.4f} | "
          f"max: {metadata['S_final'].max():.4f}")
    print(f"      Done in {time.time()-t0:.2f}s")

    # ------------------------------------------------------------------
    # Step 5: Rank and generate reasoning
    # ------------------------------------------------------------------
    print("\n[5/5] Ranking and generating reasoning ...")
    t0 = time.time()

    # Sort descending, tiebreak by candidate_id ascending (deterministic)
    ranked = (
        metadata[~disqualified]
        .sort_values(["S_final", "candidate_id"], ascending=[False, True])
        .head(100)
        .reset_index(drop=True)
    )
    ranked["rank"] = ranked.index + 1

    # Generate reasoning — data-driven, rank-aware, no templates
    ranked["reasoning"] = ranked.apply(
        lambda row: generate_reasoning(row, int(row["rank"])), axis=1
    )

    print(f"      Reasoning generated for {len(ranked)} candidates")
    print(f"      Done in {time.time()-t0:.2f}s")

    # ------------------------------------------------------------------
    # Output CSV
    # ------------------------------------------------------------------
    output = ranked[["candidate_id", "rank", "S_final", "reasoning"]].copy()
    output.columns = ["candidate_id", "rank", "score", "reasoning"]
    output["score"] = output["score"].round(6)
    output.to_csv(out_path, index=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"Ranking complete in {total_time:.1f}s")
    print(f"Output: {out_path}")
    print(f"\nTop 10 candidates:")
    print(f"{'Rank':<6} {'Candidate ID':<15} {'Score':<10} {'Title':<35} Reasoning[:60]")
    print("-" * 120)
    for _, row in ranked.head(10).iterrows():
        reasoning_preview = row["reasoning"][:60] + "..." if len(row["reasoning"]) > 60 else row["reasoning"]
        print(f"{int(row['rank']):<6} {row['candidate_id']:<15} {row['S_final']:.4f}     "
              f"{str(row['current_title']):<35} {reasoning_preview}")
    print("=" * 60)


if __name__ == "__main__":
    main()