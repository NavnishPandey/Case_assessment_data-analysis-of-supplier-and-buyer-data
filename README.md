## 📌 Task A: Supplier Data Cleaning & Joining

The datasets provided with two Excel files:

- `supplier_data1.xlsx`  
- `supplier_data2.xlsx`  

The goal is to:

1. **Clean and normalize both datasets**  
   - Unify thickness/width/weight formats  
   - Standardize categorical names (e.g., quality choice, finish, grade)  
   - Normalize column naming  

2. **Handle missing or inconsistent values**  
   - Apply column-specific strategies (median, mode, defaults, cross-fill, leave missing)  
   - Document assumptions for each decision  

3. **Join into a single inventory dataset**  
   - Align schemas from both suppliers  
   - Create `inventory_dataset.csv` with a unified schema  
   - Add metadata fields such as processing date/time, calculated weights, etc.  

4. **Document assumptions**  
   - All assumptions during cleaning are automatically logged  
   - A comprehensive cleaning report is generated (`inventory_dataset_cleaning_report.txt`)  

**Deliverables:**
- `inventory_dataset.csv` (final cleaned dataset)  
- `inventory_dataset_cleaning_report.txt` (documented assumptions, statistics, and quality metrics)

---

## 🛠️ Tools & Libraries

The pipeline is implemented in **Python** using:

- `pandas` — data manipulation and cleaning  
- `numpy` — numerical processing  
- `re` — regex parsing for text/grades  
- `logging` — process logging and assumption tracking  
- `datetime` — timestamps and metadata generation  

---

## 📑 Process & Assumptions

The cleaning pipeline (`InventoryDataProcessor` in `task_A1.py`) follows these steps:

---

### 1. Load Data
- Reads both supplier files (Excel/CSV supported).

---

### 2. Standardize Columns
- Renames columns into a unified schema.  
- Examples:  
  - `Thickness (mm)` → `thickness_mm`  
  - `Gross weight (kg)` → `weight_kg`  
  - `Finish` translated (`gebeizt` → `pickled`, `ungebeizt` → `unpickled`)  

---

### 3. Validate Numeric Columns
- Outliers beyond realistic ranges (e.g., thickness outside `0.05–100mm`) are set to `NaN`.  
- Invalid values are logged as assumptions.  

---

### 4. Standardize Categorical Values
- **Quality/Choice**: maps `"1st"`, `"first"`, `"I"` → `"1st"`  
- **Finish**: standardized to English (`oiled`, `pickled`, `bright`, etc.)  
- **Grade**: uppercased, missing values extracted from `description` if possible  

---

### 5. Handle Missing Values
Column-specific strategies:
- `thickness_mm`, `width_mm`: filled with median or defaults (`1.0 mm`, `1000 mm`)  
- `weight_kg`: filled with median or `1.0`  
- `quantity`: default `1`  
- `reserved`: default `False`  
- `grade` / `material_type`: cross-filled or `"UNKNOWN"`  
- Mechanical properties (`rp02`, `rm`, `ag`, `ai`): left missing (valid absence)  

---

### 6. Join Datasets
- Unified schema:  
    article_id, data_source, material_type, grade, quality_choice,
    finish, thickness_mm, width_mm, weight_kg, quantity,
    reserved, description, coating, has_mechanical_data,
    rp02, rm, ag, ai, total_weight_kg, available_quantity,
    processing_date, processing_time

- Ensures unique IDs (`SUP1_xxxxxx`, `SUP2_xxxxxx`) if missing  
- Calculates total weights and available quantities  

---

### 7. Final Validation
- Ensures non-negative weights and quantities  
- Removes duplicates based on `article_id + data_source`  

---

### 8. Generate Report
The file `inventory_dataset_cleaning_report.txt` includes:
- Processing summary  
- Data quality metrics  
- Remaining missing values  
- All logged assumptions  
- Column data types  

---

## 📊 Outputs

1. **`inventory_dataset.csv`**  
 - Cleaned and unified inventory dataset ready for analysis.  

2. **`inventory_dataset_cleaning_report.txt`**  
 - Comprehensive documentation of assumptions, statistics, and cleaning decisions.  


 # 📌 Task B — RFQ Similarity Analysis

The RFQ similarity pipeline (`RFQSimilarityPipeline` in `task_b.py`) performs a complete analysis of RFQ data, including reference enrichment, feature engineering, and similarity computation. The pipeline is executed via `run.py` after Task A.

---

## B.1 Reference Join & Missing Values (25 pts)

- **Grade normalization** (`task_b1.py`):  
  - Converts grades to uppercase.  
  - Strips spaces and standardizes naming.  
  - Handles suffixes and aliases.  

- **Range parsing** (`task_b1.py`):  
  - Converts numeric ranges (e.g., `"200-250"`) into `min`, `max`, and `mid`.  
  - Handles inequalities (`≤`, `≥`) and single values.  

- **Reference join** (`task_b1.py`):  
  - Joins RFQs with `reference_properties.tsv` using normalized grades.  
  - Flags RFQs with missing reference entries.  
  - Keeps missing values as `NaN` for transparency.  

- **Output**:  
  - `rfq_enriched.csv`

---

## B.2 Feature Engineering (20 pts)

Implemented in `task_b2.py`.

- **Dimensions**:  
  - Represented as intervals (`min`, `max`).  
  - If only one value is given → `min = max`.  
  - Overlap metric: *Intersection over Union (IoU)*.  

- **Categorical features**:  
  - Exact-match similarity (1/0) for `coating`, `finish`, `form`, `surface_type`.  

- **Grade properties**:  
  - Midpoints of numeric ranges from reference data are used as features.  
  - Sparse features (missing in most RFQs) are dropped.  

- **Output**:  
  - Feature-engineered dataframe passed to similarity calculation.  

---

## B.3 Similarity Calculation (30 pts)

Implemented in `task_b3.py`.

- **Aggregate similarity score**:  
  - Weighted combination of three components:  
    - Dimensions (IoU) → weight **0.4**  
    - Categorical matches → weight **0.3**  
    - Grade similarity → weight **0.3**  

- **Top-3 matches**:  
  - Each RFQ is compared against all others.  
  - Self-matches and exact duplicates are excluded.  
  - Top-3 most similar RFQs are stored.  

- **Output format** (`top3.csv`):  


---

## Bonus / Stretch Goals

- **Ablation analysis**:  
- Run similarity using only subsets of features (dimensions only, grade only).  
- Compare impact on results.  

- **Alternative metrics**:  
- Test cosine similarity or Jaccard similarity alongside IoU.  

- **Clustering**:  
- Apply clustering on engineered features to group RFQs.  
- Summarize clusters by dominant grade, form, and dimension ranges.  

---

## Pipeline Structure (`task_b.py`)

1. **Initialization**  
 - Loads `rfq.csv` and `reference_properties.tsv`.  
 - Calls `task_b1` for reference join and preprocessing.  

2. **Feature Engineering**  
 - Calls `task_b2` to transform dimensions, categorical, and grade properties into engineered features.  

3. **Similarity Calculation**  
 - Calls `task_b3` to compute similarity matrices.  
 - Aggregates similarity scores using weighted average.  
 - Extracts top-3 matches per RFQ.  

4. **Outputs**  
 - `rfq_enriched.csv` — RFQs joined with reference properties  
 - `top3.csv` — Top-3 most similar RFQs per line  

---

# 🚀 Running the Full Pipeline: Task A & Task B

This guide explains how to execute the complete pipeline for:

1. **Task A** — Supplier Inventory Data Cleaning  
2. **Task B** — RFQ Similarity Analysis  

All tasks are orchestrated via the `run.py` script.

---

## 📝 Prerequisites

- Python 3.9+ environment
- Required libraries installed:
  ```bash
  pip install requirements.txt

