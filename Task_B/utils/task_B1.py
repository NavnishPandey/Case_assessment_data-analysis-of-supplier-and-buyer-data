import pandas as pd
import re


def normalize_grade(grade: str) -> str:
    """
    Normalize grade keys:
    - Strip whitespace
    - Uppercase
    - Remove common separators like extra dashes/spaces
    """
    if pd.isna(grade):
        return None
    grade = grade.strip().upper()
    grade = re.sub(r"[^A-Z0-9]+", "", grade)  # remove non-alphanumeric
    return grade


def parse_range(value):
    """
    Parse strings like '450-600', '<=0.25', '≥250', '0.2' into numeric min/max.
    Returns dict: {min, max, mid}
    """
    if pd.isna(value):
        return {"min": None, "max": None, "mid": None}

    s = str(value).strip()

    # Range case (450-600)
    if re.match(r"^\d+(\.\d+)?-\d+(\.\d+)?$", s):
        parts = s.split("-")
        min_v, max_v = float(parts[0]), float(parts[1])
        return {"min": min_v, "max": max_v, "mid": (min_v + max_v) / 2}

    # Less than or equal
    if s.startswith("<=") or s.startswith("≤"):
        try:
            max_v = float(s.lstrip("<=≤"))
            return {"min": None, "max": max_v, "mid": max_v}
        except:
            return {"min": None, "max": None, "mid": None}

    # Greater than or equal
    if s.startswith(">=") or s.startswith("≥"):
        try:
            min_v = float(s.lstrip(">=≥"))
            return {"min": min_v, "max": None, "mid": min_v}
        except:
            return {"min": None, "max": None, "mid": None}

    # Single numeric value
    try:
        val = float(s)
        return {"min": val, "max": val, "mid": val}
    except:
        return {"min": None, "max": None, "mid": None}


def expand_reference_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand key numeric columns into min/max/mid values
    """
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


def task_b1_pipline(rfq_path, reference_path):
    # Load datasets
    rfq = pd.read_csv(rfq_path)
    reference = pd.read_csv(reference_path, sep="\t")

    # Normalize grades
    rfq["grade_norm"] = rfq["grade"].apply(normalize_grade)
    reference["grade_norm"] = reference["Grade/Material"].apply(normalize_grade)

    # Expand ranges in reference
    reference_expanded = expand_reference_ranges(reference)

    # Join on normalized grade
    merged = rfq.merge(reference_expanded, on="grade_norm", how="left", suffixes=("_rfq", "_ref"))

    # Flag missing reference joins
    merged["reference_missing"] = merged["Grade/Material"].isna()

    # Handle missing values: keep-null but flagged
    print("Merged shape:", merged.shape)
    print("Missing reference count:", merged["reference_missing"].sum())

    merged.to_csv("rfq_enriched.csv", index=False)
    print("Enriched RFQ saved -> rfq_enriched.csv")


