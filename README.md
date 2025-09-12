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


 ## 📑 Task B — RFQ Similarity Analysis

The RFQ similarity pipeline (`RFQSimilarityAnalyzer` in `task_B.py`) performs a complete analysis of RFQ data, including reference enrichment, feature engineering, similarity computation, and optional advanced analyses. The pipeline is executed via `run.py` after Task A.

---

### B.1 Reference Join & Missing Values (25 pts)

- **Normalize grade keys**:  
  - Uppercase, strip spaces, handle optional suffixes.  
  - Map aliases to standard grades (e.g., `S235`, `235JR` → `S235JR`).  

- **Parse range strings**:  
  - Convert numeric ranges (e.g., `"200-250"`) into `min`, `max`, and optional `mid`.  
  - Handle inequalities (`≤`, `≥`) and single-value strings.  

- **Join RFQs with reference properties**:  
  - Use normalized grades for joining.  
  - Handle missing reference values via flagging or median imputation.  

---

### B.2 Feature Engineering (20 pts)

- **Dimensions**:  
  - Represent each dimension as an interval (`min`, `max`).  
  - Singletons: `min = max`.  
  - Suggest overlap metric: IoU (Intersection over Union).  

- **Categorical features**:  
  - `coating`, `finish`, `form`, `surface_type` similarity = exact match (1/0).  

- **Grade properties**:  
  - Use numeric midpoints of ranges (`_mid`).  
  - Ignore sparse features if present in <20% of RFQs.  

- **Documented outputs**:  
  - `engineered_df` contains all features ready for similarity calculations.  

### B.3 Similarity Calculation (30 pts)

- **Aggregate similarity score**:  
  - Weighted combination of dimension IoU, categorical matches, and grade similarity.  
  - Default weights: dimensions 0.4, categorical 0.3, grade 0.3.  

- **Top-3 matches**:  
  - Exclude self and exact duplicates.  
  - Output CSV: `top3.csv`  
    ```
    Columns: [rfq_id, match_id, similarity_score]
    ```

---

### Bonus / Stretch Goals

- **Ablation analysis**:  
  - Compare similarity when dropping feature groups: dimensions only, grade only, etc.  
  - Adjust weights to test feature importance.  

- **Alternative metrics**:  
  - Weighted cosine + Jaccard similarity versus IoU or other approaches.  
  - Document metric definitions and results.  

- **Clustering**:  
  - Group RFQs into families using KMeans on numeric + one-hot embeddings.  
  - Provide cluster summaries with dominant grades, coatings, forms, and dimension patterns.  

---

### Pipeline Structure (`task_B.py`)

1. **Initialization**:  
   - Load RFQ CSV and reference TSV.  
   - Normalize grades and enrich RFQs with reference properties.  

2. **Data preprocessing**:  
   - Parse range strings, handle missing values, flag incomplete reference data.  

3. **Feature engineering**:  
   - Ensure dimension intervals.  
   - Encode categorical features.  
   - Keep grade properties present in ≥20% of RFQs.  

4. **Similarity matrices**:  
   - Dimension IoU  
   - Categorical exact match  
   - Grade property similarity (scaled Euclidean distance → similarity)  

5. **Aggregate similarity**:  
   - Weighted average of dimension, categorical, and grade similarity.  

6. **Generate top-3 matches**:  
   - Saved to `top3.csv`  
   - Returned as `DataFrame` for further use.  

7. **Bonus analyses**:  
   - **Ablation**: test impact of feature groups/weights.  
   - **Alternative metrics**: cosine, Jaccard, hybrid.  
   - **Clustering**: group RFQs, summarize cluster patterns.  

8. **Outputs**:
   - **top3.csv** — Top-3 similarity matches per RFQ.


## 📂 Project Structure
│
├── Task_A/
│ ├── task_A1.py # InventoryDataProcessor implementation
│ ├── pycache/ # cache files
│
├── Task_B/ # (for Scenario B, not covered here)
│ ├── task_b.py
│
├── run.py # Entry point to execute pipelines
├── README.md # Documentation 

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
  
