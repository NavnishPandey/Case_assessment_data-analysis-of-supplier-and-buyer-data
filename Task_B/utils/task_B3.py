import pandas as pd
import re
import numpy as np


def normalize_grade(grade: str) -> str:
    if pd.isna(grade):
        return None
    grade = grade.strip().upper()
    grade = re.sub(r"[^A-Z0-9]+", "", grade)
    return grade


def parse_range(value):
    if pd.isna(value):
        return {"min": None, "max": None, "mid": None}
    s = str(value).strip()
    if re.match(r"^\d+(\.\d+)?-\d+(\.\d+)?$", s):
        parts = s.split("-")
        min_v, max_v = float(parts[0]), float(parts[1])
        return {"min": min_v, "max": max_v, "mid": (min_v + max_v) / 2}
    if s.startswith("<=") or s.startswith("≤"):
        try:
            max_v = float(s.lstrip("<=≤"))
            return {"min": None, "max": max_v, "mid": max_v}
        except:
            return {"min": None, "max": None, "mid": None}
    if s.startswith(">=") or s.startswith("≥"):
        try:
            min_v = float(s.lstrip(">=≥"))
            return {"min": min_v, "max": None, "mid": min_v}
        except:
            return {"min": None, "max": None, "mid": None}
    try:
        val = float(s)
        return {"min": val, "max": val, "mid": val}
    except:
        return {"min": None, "max": None, "mid": None}


def expand_reference_ranges(df: pd.DataFrame) -> pd.DataFrame:
    range_cols = [
        "Carbon (C)", "Manganese (Mn)", "Silicon (Si)", "Sulfur (S)",
        "Phosphorus (P)", "Chromium (Cr)", "Nickel (Ni)", "Molybdenum (Mo)",
        "Vanadium (V)", "Tungsten (W)", "Copper (Cu)", "Aluminum (Al)",
        "Titanium (Ti)", "Niobium (Nb)", "Boron (B)", "Nitrogen (N)",
        "Tensile strength (Rm)", "Yield strength (Re or Rp0.2)", "Elongation (A%)"
    ]
    for col in range_cols:
        if col in df.columns:
            parsed = df[col].apply(parse_range)
            df[f"{col}_min"] = parsed.apply(lambda x: x["min"])
            df[f"{col}_max"] = parsed.apply(lambda x: x["max"])
            df[f"{col}_mid"] = parsed.apply(lambda x: x["mid"])
    return df


def interval_iou(min1, max1, min2, max2):
    if pd.isna(min1) or pd.isna(max1) or pd.isna(min2) or pd.isna(max2):
        return None
    if max1 < min2 or max2 < min1:
        return 0.0
    overlap = max(0, min(max1, max2) - max(min1, min2))
    union = max(max1, max2) - min(min1, min2)
    return overlap / union if union > 0 else 0.0


def feature_engineering(merged: pd.DataFrame) -> pd.DataFrame:
    dim_pairs = [
        ("thickness_min", "thickness_max"),
        ("width_min", "width_max"),
        ("height_min", "height_max"),
        ("weight_min", "weight_max"),
        ("inner_diameter_min", "inner_diameter_max"),
        ("outer_diameter_min", "outer_diameter_max"),
        ("yield_strength_min", "yield_strength_max"),
        ("tensile_strength_min", "tensile_strength_max"),
    ]
    for min_col, max_col in dim_pairs:
        if min_col in merged.columns and max_col in merged.columns:
            merged[f"{min_col}_vs_ref_iou"] = merged.apply(
                lambda row: interval_iou(
                    row[min_col], row[max_col],
                    row.get(f"{min_col}_ref"), row.get(f"{max_col}_ref")
                ), axis=1
            )
    cat_cols = ["coating", "finish", "form", "surface_type"]
    for col in cat_cols:
        if f"{col}_ref" in merged.columns and f"{col}_rfq" in merged.columns:
            merged[f"{col}_similarity"] = (
                merged[f"{col}_rfq"].fillna("") == merged[f"{col}_ref"].fillna("")
            ).astype(int)
    grade_mid_cols = [c for c in merged.columns if c.endswith("_mid")]
    for col in grade_mid_cols:
        merged[f"feature_{col}"] = merged[col]
    sparse_cols = [c for c in merged.columns if merged[c].isna().mean() > 0.9]
    merged = merged.drop(columns=sparse_cols)
    return merged


def grade_similarity(v1, v2):
    if pd.isna(v1) or pd.isna(v2):
        return None
    return 1 - abs(v1 - v2) / max(v1, v2) if max(v1, v2) != 0 else 1


def compute_similarity(df: pd.DataFrame):
    dim_cols = [c for c in df.columns if c.endswith("_vs_ref_iou")]
    cat_cols = [c for c in df.columns if c.endswith("_similarity")]
    grade_cols = [c for c in df.columns if c.startswith("feature_")]

    results = []
    for i, row_i in df.iterrows():
        sims = []
        for j, row_j in df.iterrows():
            if i == j:
                continue
            # Dimensions similarity (mean of IoUs)
            dim_scores = [interval_iou(row_i.get(dc.replace("_vs_ref", "_rfq")),
                                       row_i.get(dc.replace("_vs_ref", "_rfq").replace("_min","_max")),
                                       row_j.get(dc.replace("_vs_ref", "_rfq")),
                                       row_j.get(dc.replace("_vs_ref", "_rfq").replace("_min","_max")))
                           for dc in dim_cols if not pd.isna(row_i.get(dc.replace("_vs_ref", "_rfq")))]
            dim_score = np.nanmean(dim_scores) if dim_scores else 0

            # Categorical similarity
            cat_scores = [1 if row_i.get(c.replace("_similarity", "_rfq")) == row_j.get(c.replace("_similarity", "_rfq")) else 0
                          for c in cat_cols]
            cat_score = np.mean(cat_scores) if cat_scores else 0

            # Grade similarity
            grade_scores = [grade_similarity(row_i[g], row_j[g]) for g in grade_cols]
            grade_score = np.nanmean([s for s in grade_scores if s is not None]) if grade_scores else 0

            # Weighted aggregate
            agg_score = 0.4 * dim_score + 0.3 * cat_score + 0.3 * grade_score
            sims.append((j, agg_score))
        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
        for rank, (j, score) in enumerate(sims_sorted, start=1):
            results.append({"rfq_id": row_i["id"], "similar_rfq_id": df.loc[j, "id"], "rank": rank, "similarity": score})
    return pd.DataFrame(results)


def task_b3_pipline(rfq_path, reference_path):
    rfq = pd.read_csv(rfq_path)
    reference = pd.read_csv(reference_path, sep="\t")

    rfq["grade_norm"] = rfq["grade"].apply(normalize_grade)
    reference["grade_norm"] = reference["Grade/Material"].apply(normalize_grade)

    reference_expanded = expand_reference_ranges(reference)
    merged = rfq.merge(reference_expanded, on="grade_norm", how="left", suffixes=("_rfq", "_ref"))
    merged["reference_missing"] = merged["Grade/Material"].isna()

    enriched = feature_engineering(merged)

    sim_df = compute_similarity(enriched)
    sim_df.to_csv("top3.csv", index=False)
    print("Top-3 similar RFQs per line saved -> rfq_similarity.csv")


