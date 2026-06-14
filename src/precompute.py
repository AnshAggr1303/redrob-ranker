"""
precompute.py — Phase 1 offline pre-computation
Redrob Intelligent Candidate Ranking System

Run once locally. No time limit. Produces four artefacts in ./artefacts/:
  - embeddings.npy       float32 (N, 384) — candidate narrative embeddings
  - jd_embedding.npy     float32 (384,)   — signal-only JD embedding
  - metadata.parquet     DataFrame         — structured fields for scoring
  - honeypot_ids.txt     one CAND_XXXXXXX per line

Usage:
    python precompute.py --candidates ./data/candidates.jsonl \
                         --jd ./data/signal_jd.txt \
                         --out ./artefacts

Dependencies:
    pip install sentence-transformers numpy pandas pyarrow tqdm
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 512

# Consulting firm patterns — normalized regex (case-insensitive)
# Strips common suffixes before matching
CONSULTING_PATTERNS = [
    r"\btcs\b",
    r"\binfosys\b",
    r"\bwipro\b",
    r"\baccenture\b",
    r"\bcognizant\b",
    r"\bcapgemini\b",
    r"\bhcl\b",
    r"\btech\s*mahindra\b",
]
CONSULTING_RE = [re.compile(p, re.IGNORECASE) for p in CONSULTING_PATTERNS]

# Suffixes to strip before company name matching
COMPANY_SUFFIX_RE = re.compile(
    r"\b(ltd\.?|limited|technologies|tech|solutions|services|india|pvt\.?|private|inc\.?|corp\.?|llc)\b",
    re.IGNORECASE,
)

# Relevant skill assessments for this JD
RELEVANT_ASSESSMENTS = {
    "nlp", "python", "machine learning", "information retrieval",
    "embeddings", "recommendation systems", "search", "deep learning",
    "vector search", "ranking", "retrieval", "transformers",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_company(name: str) -> str:
    """Lowercase, strip common suffixes, collapse whitespace."""
    name = name.lower()
    name = COMPANY_SUFFIX_RE.sub(" ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def is_consulting(company_name: str) -> bool:
    normalized = normalize_company(company_name)
    return any(p.search(normalized) for p in CONSULTING_RE)


def build_embedding_text(candidate: dict) -> str:
    """
    Concatenate headline + summary + all career descriptions.
    No skills, no certifications — narrative only.
    """
    profile = candidate["profile"]
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
    ]
    for role in candidate.get("career_history", []):
        desc = role.get("description", "")
        if desc:
            parts.append(desc)
    return " ".join(p.strip() for p in parts if p.strip())


def detect_honeypot(candidate: dict) -> list[str]:
    """
    Return list of honeypot flag strings. Empty list = clean candidate.

    Rules:
    1. Inverted salary (min > max)
    2. Any skill duration_months > total_career_months + 6
    3. Claimed YOE vs actual career span: total_career_months < YOE * 12 * 0.5
    4. Expert skill with duration_months == 0
    """
    flags = []
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Rule 1 — inverted salary
    sal = signals.get("expected_salary_range_inr_lpa", {})
    if sal.get("min") is not None and sal.get("max") is not None:
        if sal["min"] > sal["max"]:
            flags.append(f"INVERTED_SALARY:{sal['min']:.1f}>{sal['max']:.1f}")

    # Total career months
    total_months = sum(r.get("duration_months", 0) for r in career)

    # Rule 2 — skill duration overflow
    for sk in skills:
        sk_months = sk.get("duration_months", 0)
        if sk_months > total_months + 6:
            flags.append(f"SKILL_OVERFLOW:{sk['name']}:{sk_months}mo>career:{total_months}mo")

    # Rule 3 — YOE vs career span mismatch
    yoe = profile.get("years_of_experience", 0) or 0
    if total_months < (yoe * 12) * 0.5 and yoe > 1:
        flags.append(f"YOE_MISMATCH:claims_{yoe}yr_but_{total_months}mo_career")

    # Rule 4 — expert skill with 0 duration
    for sk in skills:
        if sk.get("proficiency") == "expert" and sk.get("duration_months", 0) == 0:
            flags.append(f"EXPERT_ZERO_DURATION:{sk['name']}")

    return flags


def extract_metadata(candidate: dict) -> dict:
    """Extract all structured fields needed at ranking time."""
    cid = candidate["candidate_id"]
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})

    # Career companies and industries
    career_companies = [r.get("company", "") for r in career]
    career_industries = list({r.get("industry", "") for r in career if r.get("industry")})
    total_career_months = sum(r.get("duration_months", 0) for r in career)

    # Consulting flags
    all_consulting = (
        len(career_companies) > 0
        and all(is_consulting(c) for c in career_companies)
    )
    any_consulting = any(is_consulting(c) for c in career_companies)
    has_mixed_consulting = any_consulting and not all_consulting

    # Education best tier
    tier_order = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 5}
    edu_tiers = [e.get("tier", "unknown") for e in education]
    best_tier = min(edu_tiers, key=lambda t: tier_order.get(t, 5)) if edu_tiers else "unknown"

    # Skill assessment scores — check for relevant ones
    assess_scores = signals.get("skill_assessment_scores", {}) or {}
    relevant_assess_score = max(
        (v for k, v in assess_scores.items() if k.lower() in RELEVANT_ASSESSMENTS),
        default=0.0,
    )

    # Last active date as string (converted to datetime in rank.py)
    last_active = signals.get("last_active_date", "2020-01-01")

    return {
        "candidate_id": cid,
        "years_of_experience": profile.get("years_of_experience", 0) or 0,
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "current_industry": profile.get("current_industry", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "willing_to_relocate": signals.get("willing_to_relocate", False),
        "career_companies_json": json.dumps(career_companies),
        "career_industries_json": json.dumps(career_industries),
        "total_career_months": total_career_months,
        "all_consulting": all_consulting,
        "has_mixed_consulting": has_mixed_consulting,
        "best_education_tier": best_tier,
        "relevant_assess_score": relevant_assess_score,
        # redrob_signals
        "last_active_date": last_active,
        "open_to_work_flag": signals.get("open_to_work_flag", False),
        "notice_period_days": signals.get("notice_period_days", 90),
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0.5),
        "interview_completion_rate": signals.get("interview_completion_rate", 0.5),
        "offer_acceptance_rate": signals.get("offer_acceptance_rate", -1),
        "avg_response_time_hours": signals.get("avg_response_time_hours", 48),
        "github_activity_score": signals.get("github_activity_score", -1),
        "profile_completeness_score": signals.get("profile_completeness_score", 0),
        "verified_email": signals.get("verified_email", False),
        "verified_phone": signals.get("verified_phone", False),
        "linkedin_connected": signals.get("linkedin_connected", False),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Redrob Phase 1 Pre-computation")
    parser.add_argument("--candidates", default="./data/candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--jd", default="./data/signal_jd.txt",
                        help="Path to signal_jd.txt")
    parser.add_argument("--out", default="./artefacts",
                        help="Output directory for artefacts")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load all candidates
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading candidates from {args.candidates} ...")
    t0 = time.time()
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"      Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Honeypot detection
    # ------------------------------------------------------------------
    print(f"\n[2/5] Running honeypot detection ...")
    t0 = time.time()
    honeypot_ids = set()
    honeypot_log = []

    for c in tqdm(candidates, desc="Scanning", unit="cand"):
        flags = detect_honeypot(c)
        if flags:
            honeypot_ids.add(c["candidate_id"])
            honeypot_log.append({"id": c["candidate_id"], "flags": flags})

    honeypot_path = out_dir / "honeypot_ids.txt"
    with open(honeypot_path, "w") as f:
        for cid in sorted(honeypot_ids):
            f.write(cid + "\n")

    print(f"      Found {len(honeypot_ids):,} honeypots in {time.time()-t0:.1f}s")
    print(f"      Written to {honeypot_path}")

    # Print sample flags for verification
    for entry in honeypot_log[:5]:
        print(f"      {entry['id']}: {entry['flags']}")

    # ------------------------------------------------------------------
    # Step 3: Extract metadata → parquet
    # ------------------------------------------------------------------
    print(f"\n[3/5] Extracting structured metadata ...")
    t0 = time.time()
    rows = [extract_metadata(c) for c in tqdm(candidates, desc="Extracting", unit="cand")]
    df = pd.DataFrame(rows)
    meta_path = out_dir / "metadata.parquet"
    df.to_parquet(meta_path, index=False)
    print(f"      Metadata shape: {df.shape} → {meta_path}")
    print(f"      Done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Build candidate embeddings
    # ------------------------------------------------------------------
    print(f"\n[4/5] Building candidate embeddings (model: {EMBEDDING_MODEL}) ...")
    print(f"      This will take a few minutes for 100k candidates ...")
    t0 = time.time()

    model = SentenceTransformer(EMBEDDING_MODEL)
    model = model.to("mps")

    texts = [build_embedding_text(c) for c in candidates]

    # Encode in batches with progress bar
    all_embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,   # we normalize in rank.py for cosine sim
    ).astype(np.float32)

    emb_path = out_dir / "embeddings.npy"
    np.save(emb_path, all_embeddings.astype(np.float32))
    print(f"      Embeddings shape: {all_embeddings.shape} → {emb_path}")
    print(f"      Done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 5: JD embedding
    # ------------------------------------------------------------------
    print(f"\n[5/5] Building JD embedding from {args.jd} ...")
    t0 = time.time()

    with open(args.jd, "r", encoding="utf-8") as f:
        jd_text = f.read().strip()

    jd_embedding = model.encode(
        [jd_text],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )[0].astype(np.float32)

    jd_path = out_dir / "jd_embedding.npy"
    np.save(jd_path, jd_embedding.astype(np.float32))
    print(f"      JD embedding shape: {jd_embedding.shape} → {jd_path}")
    print(f"      Done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Pre-computation complete. Artefacts:")
    for fname in ["embeddings.npy", "jd_embedding.npy", "metadata.parquet", "honeypot_ids.txt"]:
        p = out_dir / fname
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {fname:<25} {size_mb:>8.1f} MB")
    print("="*60)
    print("\nReady to run rank.py\n")


if __name__ == "__main__":
    main()