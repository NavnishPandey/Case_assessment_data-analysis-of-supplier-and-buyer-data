import numpy as np
import pandas as pd
import re
from typing import Tuple, List, Optional, Dict
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class RFQSimilarityAnalyzer:
    """Complete RFQ similarity analysis with console output for all results except top3."""
    
    def __init__(self, rfq_file: str, reference_file: str):
        self.rfq_file = rfq_file
        self.reference_file = reference_file
        self.enriched_df = None
        self.engineered_df = None
        self.grade_property_cols = []
        
    @staticmethod
    def normalize_grade_key(grade: str, suffix: str = None) -> Optional[str]:
        """Normalize grade key with optional suffix."""
        if pd.isna(grade):
            return None
        normalized = str(grade).upper().strip()
        if suffix and not pd.isna(suffix):
            suffix_str = str(suffix).upper().strip()
            suffix_str = re.sub(r'^[^A-Z0-9]+|[^A-Z0-9]+$', '', suffix_str)
            if suffix_str:
                normalized = f"{normalized}{suffix_str}"
        aliases = {
            'S235JR': ['S235', '235JR'],
            'S355JR': ['S355', '355JR'],
            'S275JR': ['S275', '275JR'],
        }
        for standard, alias_list in aliases.items():
            if normalized in alias_list:
                return standard
        return normalized

    @staticmethod
    def parse_range_string(range_str, is_percentage=False):
        """Parse range string into min, max, and midpoint values."""
        if pd.isna(range_str) or str(range_str).strip() in ['', 'NaN', 'None']:
            return None, None, None
        rs = str(range_str).strip()
        m = re.search(r'(\d*\.?\d+)\s*[-–]\s*(\d*\.?\d+)', rs)
        if m:
            mn = float(m.group(1)); mx = float(m.group(2)); mid = (mn + mx) / 2
            return mn, mx, mid
        m2 = re.search(r'[≤≤]\s*(\d*\.?\d+)', rs)
        if m2:
            mx = float(m2.group(1)); return None, mx, (mx/2 if mx else None)
        m3 = re.search(r'[≥≥]\s*(\d*\.?\d+)', rs)
        if m3:
            mn = float(m3.group(1)); return mn, None, (mn*1.5 if mn else None)
        m4 = re.search(r'^(\d*\.?\d+)$', rs)
        if m4:
            v = float(m4.group(1)); return v, v, v
        if is_percentage and '%' in rs:
            num_match = re.search(r'(\d*\.?\d+)', rs)
            if num_match:
                v = float(num_match.group(1)); return v, v, v
        return None, None, None

    def load_and_preprocess_data(self) -> None:
        """Load and preprocess the RFQ and reference data."""
        print("Loading input files...")
        rfq_df = pd.read_csv(self.rfq_file)
        ref_df = pd.read_csv(self.reference_file, sep='\t')

        print(f"RFQ rows: {len(rfq_df)}, Reference rows: {len(ref_df)}")

        # Normalize grades
        rfq_df['normalized_grade'] = rfq_df.apply(
            lambda r: self.normalize_grade_key(r.get('grade'), r.get('grade_suffix', None)), axis=1
        )
        ref_df['normalized_grade_ref'] = ref_df['Grade/Material'].apply(
            lambda x: self.normalize_grade_key(x)
        )

        candidate_range_cols = [
            'Carbon (C)', 'Manganese (Mn)', 'Silicon (Si)', 'Sulfur (S)',
            'Phosphorus (P)', 'Chromium (Cr)', 'Nickel (Ni)', 'Molybdenum (Mo)',
            'Vanadium (V)', 'Tungsten (W)', 'Cobalt (Co)', 'Copper (Cu)',
            'Aluminum (Al)', 'Titanium (Ti)', 'Niobium (Nb)', 'Boron (B)',
            'Nitrogen (N)', 'Tensile strength (Rm)', 'Yield strength (Re or Rp0.2)',
            'Elongation (A%)', 'Reduction of area (Z%)'
        ]
        existing = [c for c in candidate_range_cols if c in ref_df.columns]
        for col in existing:
            is_pct = '%' in col or col in ['Elongation (A%)', 'Reduction of area (Z%)']
            parsed = ref_df[col].apply(lambda x: self.parse_range_string(x, is_pct))
            ref_df[f'{col}_min'] = parsed.apply(lambda t: t[0] if t else None)
            ref_df[f'{col}_max'] = parsed.apply(lambda t: t[1] if t else None)
            ref_df[f'{col}_mid'] = parsed.apply(lambda t: t[2] if t else None)

        # Join
        joined = pd.merge(
            rfq_df, ref_df, 
            left_on='normalized_grade', 
            right_on='normalized_grade_ref', 
            how='left',
            suffixes=('_rfq', '_ref')
        )

        # Imputation for key mechanical properties
        joined['reference_data_missing'] = joined['normalized_grade_ref'].isna()
        key_props = ['Tensile strength (Rm)_mid', 'Yield strength (Re or Rp0.2)_mid', 'Elongation (A%)_mid']
        
        for prop in key_props:
            if prop in joined.columns:
                joined[f'{prop}_missing'] = joined[prop].isna()
                if 'Category' in joined.columns:
                    med_by_cat = joined.groupby('Category')[prop].transform('median')
                    joined[prop] = joined[prop].fillna(med_by_cat)
                joined[prop] = joined[prop].fillna(joined[prop].median())

        self.enriched_df = joined
        print("Data loading and preprocessing completed.")

    def engineer_features(self) -> None:
        """Engineer features for similarity analysis."""
        if self.enriched_df is None:
            raise ValueError("Please run load_and_preprocess_data() first.")
            
        print("Starting feature engineering...")
        df = self.enriched_df.copy()
        
        # Ensure dimension pairs have both min and max
        dimension_pairs = [
            ('thickness_min', 'thickness_max'),
            ('width_min', 'width_max'),
            ('length_min', 'length_max'),
            ('height_min', 'height_max'),
            ('weight_min', 'weight_max'),
            ('inner_diameter_min', 'inner_diameter_max'),
            ('outer_diameter_min', 'outer_diameter_max')
        ]
        for mn, mx in dimension_pairs:
            if mn in df.columns and mx not in df.columns:
                df[mx] = df[mn]
            if mx in df.columns and mn not in df.columns:
                df[mn] = df[mx]
        
        # Handle categorical features
        categorical_columns = ['coating', 'finish', 'form', 'surface_type', 'surface_protection', 'grade']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("##NA##")
        
        # Process grade properties
        grade_property_columns = [c for c in df.columns if c.endswith('_mid')]
        cols_to_keep = []
        for c in grade_property_columns:
            ratio = df[c].notna().mean()
            if ratio >= 0.2:
                cols_to_keep.append(c)
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        
        self.engineered_df = df
        self.grade_property_cols = cols_to_keep
        print(f"Feature engineering completed. Kept {len(cols_to_keep)} grade properties.")

    def build_dimension_iou_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Build dimension IoU similarity matrix."""
        dimension_pairs = [
            ('thickness_min', 'thickness_max'),
            ('width_min', 'width_max'),
            ('length_min', 'length_max'),
            ('height_min', 'height_max'),
            ('weight_min', 'weight_max'),
            ('inner_diameter_min', 'inner_diameter_max'),
            ('outer_diameter_min', 'outer_diameter_max')
        ]
        
        n = len(df)
        sims = []
        
        for mn, mx in dimension_pairs:
            if mn in df.columns and mx in df.columns:
                # Vectorized IoU calculation
                a_min = df[mn].to_numpy().astype(float)[:, None]
                a_max = df[mx].to_numpy().astype(float)[:, None]
                b_min = df[mn].to_numpy().astype(float)[None, :]
                b_max = df[mx].to_numpy().astype(float)[None, :]
                
                inter_min = np.maximum(a_min, b_min)
                inter_max = np.minimum(a_max, b_max)
                inter_len = np.maximum(0.0, inter_max - inter_min)
                union_len = np.maximum(1e-9, (a_max - a_min) + (b_max - b_min) - inter_len)
                
                iou = inter_len / union_len
                sims.append(iou)
        
        if not sims:
            return np.zeros((n, n))
        
        # Average IoU across all dimension types
        avg_iou = np.mean(np.stack(sims, axis=0), axis=0)
        return avg_iou

    def build_categorical_match_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Build categorical match similarity matrix."""
        categorical_cols = ['coating', 'finish', 'form', 'surface_type', 'surface_protection']
        n = len(df)
        sims = []
        
        for col in categorical_cols:
            if col in df.columns:
                arr = df[col].astype(str).to_numpy()
                match = (arr[:, None] == arr[None, :]).astype(float)
                sims.append(match)
        
        if not sims:
            return np.zeros((n, n))
        
        # Average match score across all categorical features
        avg_match = np.mean(np.stack(sims, axis=0), axis=0)
        return avg_match

    def build_grade_similarity_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Build grade property similarity matrix."""
        if not self.grade_property_cols:
            n = len(df)
            return np.zeros((n, n))
        
        data = df[self.grade_property_cols].to_numpy().astype(float)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Convert Euclidean distance to similarity (0-1)
        dists = euclidean_distances(data_scaled, data_scaled)
        maxd = dists.max() if dists.max() > 0 else 1.0
        sim = 1.0 - (dists / maxd)
        sim = np.clip(sim, 0.0, 1.0)
        
        return sim

    def calculate_aggregate_similarity(self, weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> np.ndarray:
        """Calculate aggregate similarity score using weighted average."""
        if self.engineered_df is None:
            raise ValueError("Please run engineer_features() first.")
            
        df = self.engineered_df
        
        # Build individual similarity matrices
        dim_sim = self.build_dimension_iou_matrix(df)
        cat_sim = self.build_categorical_match_matrix(df)
        grade_sim = self.build_grade_similarity_matrix(df)
        
        # Weighted average
        w_dim, w_cat, w_grade = weights
        aggregate_sim = (
            w_dim * dim_sim + 
            w_cat * cat_sim + 
            w_grade * grade_sim
        )
        
        return aggregate_sim

    def generate_top3_matches(self, output_file: str = "top3.csv") -> pd.DataFrame:
        """Generate top-3 most similar RFQs for each RFQ and save to CSV."""
        if self.engineered_df is None:
            raise ValueError("Please run engineer_features() first.")
            
        # Get RFQ IDs
        rfq_ids = (self.engineered_df['id'].to_numpy() 
                  if 'id' in self.engineered_df.columns 
                  else np.arange(len(self.engineered_df)))
        
        # Calculate aggregate similarity
        aggregate_sim = self.calculate_aggregate_similarity()
        n = len(aggregate_sim)
        
        # Find top-3 matches (excluding self)
        results = []
        for i in range(n):
            # Set self-similarity to -1 to exclude
            similarities = aggregate_sim[i].copy()
            similarities[i] = -1.0
            
            # Get top-3 indices
            top_indices = np.argpartition(-similarities, range(min(3, n-1)))[:3]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
            
            # Add to results
            for j in top_indices:
                results.append({
                    'rfq_id': rfq_ids[i],
                    'match_id': rfq_ids[j],
                    'similarity_score': float(similarities[j])
                })
        
        # Create and save results
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        print(f"Generated top-3 matches: {output_file}")
        
        return df_results

    def run_ablation_analysis(self) -> pd.DataFrame:
        """Run ablation analysis by dropping feature groups and adjusting weights."""
        if self.engineered_df is None:
            raise ValueError("Please run engineer_features() first.")
            
        print("\n" + "="*60)
        print("ABLATION ANALYSIS: Feature Group Importance")
        print("="*60)
        
        df = self.engineered_df
        n = len(df)
        
        # Build individual similarity matrices
        dim_sim = self.build_dimension_iou_matrix(df)
        cat_sim = self.build_categorical_match_matrix(df)
        grade_sim = self.build_grade_similarity_matrix(df)
        
        ablation_results = []
        
        # Single feature group experiments
        experiments = [
            ('dimensions_only', (1.0, 0.0, 0.0)),
            ('categorical_only', (0.0, 1.0, 0.0)),
            ('grade_only', (0.0, 0.0, 1.0)),
            ('default_weights', (0.4, 0.3, 0.3)),
            ('dimensions_heavy', (0.6, 0.2, 0.2)),
            ('categorical_heavy', (0.2, 0.6, 0.2)),
            ('grade_heavy', (0.2, 0.2, 0.6)),
            ('balanced', (0.33, 0.33, 0.34))
        ]
        
        for exp_name, weights in experiments:
            w_dim, w_cat, w_grade = weights
            agg_sim = w_dim * dim_sim + w_cat * cat_sim + w_grade * grade_sim
            
            # Calculate average top-3 similarity
            top3_scores = []
            for i in range(n):
                sim_row = agg_sim[i].copy()
                sim_row[i] = -1  # Exclude self
                top3_idx = np.argpartition(-sim_row, 3)[:3]
                top3_scores.extend(sim_row[top3_idx])
            
            avg_top3 = np.mean(top3_scores) if top3_scores else 0
            std_top3 = np.std(top3_scores) if top3_scores else 0
            
            ablation_results.append({
                'experiment': exp_name,
                'weight_dimensions': w_dim,
                'weight_categorical': w_cat,
                'weight_grade': w_grade,
                'avg_top3_similarity': round(avg_top3, 4),
                'std_top3_similarity': round(std_top3, 4)
            })
        
        df_ablation = pd.DataFrame(ablation_results)
        
        print("\nAblation Results (Higher average similarity is better):")
        print("-" * 80)
        for _, row in df_ablation.iterrows():
            print(f"{row['experiment']:20} | Dim: {row['weight_dimensions']} | Cat: {row['weight_categorical']} | "
                  f"Grade: {row['weight_grade']} | Avg: {row['avg_top3_similarity']:.4f} ± {row['std_top3_similarity']:.4f}")
        
        
        return df_ablation

    def build_embedding(self) -> np.ndarray:
        """Build numeric embedding for alternative metrics."""
        if self.engineered_df is None:
            raise ValueError("Please run engineer_features() first.")
            
        df = self.engineered_df
        dimension_pairs = [
            ('thickness_min', 'thickness_max'),
            ('width_min', 'width_max'),
            ('length_min', 'length_max'),
            ('height_min', 'height_max'),
            ('weight_min', 'weight_max'),
            ('inner_diameter_min', 'inner_diameter_max'),
            ('outer_diameter_min', 'outer_diameter_max')
        ]
        categorical_cols = ['coating', 'finish', 'form', 'surface_type', 'surface_protection']
        
        numeric_features = []
        
        for mn, mx in dimension_pairs:
            if mn in df.columns and mx in df.columns:
                mid = (pd.to_numeric(df[mn], errors='coerce').fillna(0) + 
                      pd.to_numeric(df[mx], errors='coerce').fillna(0)) / 2
                numeric_features.append(mid.values.reshape(-1, 1))
        
        if self.grade_property_cols:
            grade_data = df[self.grade_property_cols].values
            numeric_features.append(grade_data)
        
        # Combine numeric features
        if numeric_features:
            numeric_matrix = np.hstack(numeric_features)
        else:
            numeric_matrix = np.zeros((len(df), 0))
        
        # Categorical features (one-hot encoded)
        existing_cats = [c for c in categorical_cols if c in df.columns]
        if existing_cats:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            cat_matrix = ohe.fit_transform(df[existing_cats].astype(str).fillna("##NA##"))
        else:
            cat_matrix = np.zeros((len(df), 0))
        
        # Combine all features
        if numeric_matrix.shape[1] > 0:
            scaler = StandardScaler()
            numeric_scaled = scaler.fit_transform(numeric_matrix)
            embedding = np.hstack([numeric_scaled, cat_matrix])
        else:
            embedding = cat_matrix
        
        return embedding

    def run_alternative_metrics(self) -> pd.DataFrame:
        """Compare alternative similarity metrics and display results."""
        if self.engineered_df is None:
            raise ValueError("Please run engineer_features() first.")
            
        print("\n" + "="*60)
        print("ALTERNATIVE METRICS COMPARISON")
        print("="*60)
        
        df = self.engineered_df
        n = len(df)
        
        embedding = self.build_embedding()
        
        dim_sim = self.build_dimension_iou_matrix(df)
        cat_sim = self.build_categorical_match_matrix(df)
        grade_sim = self.build_grade_similarity_matrix(df)
        
        metrics_results = []
        
        iou_agg = 0.4 * dim_sim + 0.3 * cat_sim + 0.3 * grade_sim
        
        cos_sim = cosine_similarity(embedding)
        
        categorical_cols = ['coating', 'finish', 'form', 'surface_type', 'surface_protection']
        existing_cats = [c for c in categorical_cols if c in df.columns]
        
        if existing_cats:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            bin_mat = ohe.fit_transform(df[existing_cats].astype(str).fillna("##NA##"))
            inter = bin_mat @ bin_mat.T
            row_sums = bin_mat.sum(axis=1).reshape(-1, 1)
            union = row_sums + row_sums.T - inter
            jaccard_sim = np.where(union > 0, inter / union, 0)
        else:
            jaccard_sim = np.zeros((n, n))
        
        hybrid_sim = 0.6 * cos_sim + 0.4 * jaccard_sim
        
        metrics = [
            ('iou_aggregate', iou_agg, 'Weighted average of dimension IoU, categorical matches, and grade similarity'),
            ('cosine_embedding', cos_sim, 'Cosine similarity on combined numeric and one-hot encoded features'),
            ('jaccard_categorical', jaccard_sim, 'Jaccard similarity based only on categorical feature matches'),
            ('hybrid_cosine_jaccard', hybrid_sim, 'Weighted combination of cosine and jaccard similarities (0.6:0.4)')
        ]
        
        for metric_name, sim_matrix, description in metrics:
            # Calculate average top-3 similarity
            top3_scores = []
            for i in range(n):
                sim_row = sim_matrix[i].copy()
                sim_row[i] = -1  # Exclude self
                top3_idx = np.argpartition(-sim_row, 3)[:3]
                top3_scores.extend(sim_row[top3_idx])
            
            avg_top3 = np.mean(top3_scores) if top3_scores else 0
            std_top3 = np.std(top3_scores) if top3_scores else 0
            
            metrics_results.append({
                'metric': metric_name,
                'avg_top3_similarity': round(avg_top3, 4),
                'std_top3_similarity': round(std_top3, 4),
                'description': description
            })
        
        df_metrics = pd.DataFrame(metrics_results)
        
        # Display results in console
        print("\nAlternative Metrics Performance:")
        print("-" * 80)
        for _, row in df_metrics.iterrows():
            print(f"{row['metric']:20} | Avg: {row['avg_top3_similarity']:.4f} ± {row['std_top3_similarity']:.4f}")
        
        print("\nMetric Descriptions:")
        print("-" * 80)
        for _, row in df_metrics.iterrows():
            print(f"• {row['metric']}: {row['description']}")
        
        return df_metrics

    def run_clustering(self, n_clusters: int = 5) -> Dict[str, pd.DataFrame]:
        """Cluster RFQs into families and display results in console."""
        if self.engineered_df is None:
            raise ValueError("Please run engineer_features() first.")
            
        print("\n" + "="*60)
        print("CLUSTERING ANALYSIS: RFQ Family Groups")
        print("="*60)
        
        df = self.engineered_df.copy()
        embedding = self.build_embedding()
        
        # Determine optimal number of clusters
        k = min(n_clusters, max(2, len(df) // 5))
        
        if len(df) >= k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embedding)
        else:
            clusters = np.zeros(len(df), dtype=int)
        
        df['cluster'] = clusters
        
        cluster_summaries = []
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            
            summary = {
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_data),
                'most_common_grade': cluster_data['normalized_grade'].mode().iloc[0] if 'normalized_grade' in cluster_data.columns and not cluster_data['normalized_grade'].mode().empty else 'N/A',
                'most_common_coating': cluster_data['coating'].mode().iloc[0] if 'coating' in cluster_data.columns and not cluster_data['coating'].mode().empty else 'N/A',
                'most_common_form': cluster_data['form'].mode().iloc[0] if 'form' in cluster_data.columns and not cluster_data['form'].mode().empty else 'N/A',
            }
            
            dimension_cols = ['thickness', 'width', 'length', 'weight']
            for dim in dimension_cols:
                min_col = f'{dim}_min'
                max_col = f'{dim}_max'
                if min_col in cluster_data.columns and max_col in cluster_data.columns:
                    avg_mid = (cluster_data[min_col].mean() + cluster_data[max_col].mean()) / 2
                    summary[f'avg_{dim}'] = round(avg_mid, 2)
            
            interpretation = self._interpret_cluster(cluster_data)
            summary['interpretation'] = interpretation
            
            cluster_summaries.append(summary)
        
        print("\nCLUSTER ASSIGNMENTS:")
        print("-" * 80)
        cluster_assignments = []
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_rfqs = df[df['cluster'] == cluster_id]
            rfq_ids = cluster_rfqs['id'].tolist() if 'id' in cluster_rfqs.columns else list(cluster_rfqs.index)
            cluster_assignments.append({
                'cluster': cluster_id,
                'rfq_count': len(cluster_rfqs),
                'rfq_ids': rfq_ids[:10] + ['...'] if len(rfq_ids) > 10 else rfq_ids
            })
        
        for assignment in cluster_assignments:
            print(f"Cluster {assignment['cluster']}: {assignment['rfq_count']} RFQs")
            print(f"  RFQ IDs: {assignment['rfq_ids']}")
        
        # Display cluster results
        print("\nCLUSTER SUMMARIES:")
        print("-" * 80)
        for summary in cluster_summaries:
            print(f"\nCluster {summary['cluster_id']} (Size: {summary['cluster_size']})")
            print(f"  Primary Grade: {summary['most_common_grade']}")
            print(f"  Primary Coating: {summary['most_common_coating']}")
            print(f"  Primary Form: {summary['most_common_form']}")
            
            # Show dimensions if available
            dim_info = []
            for dim in ['thickness', 'width', 'length', 'weight']:
                dim_key = f'avg_{dim}'
                if dim_key in summary:
                    dim_info.append(f"{dim}: {summary[dim_key]}")
            
            if dim_info:
                print(f"  Average Dimensions: {', '.join(dim_info)}")
            
            print(f"  Interpretation: {summary['interpretation']}")
        
        return {
            'cluster_assignments': pd.DataFrame(cluster_assignments),
            'cluster_summaries': pd.DataFrame(cluster_summaries)
        }

    def _interpret_cluster(self, cluster_data: pd.DataFrame) -> str:
        """Generate interpretation for a cluster."""
        interpretations = []
        
        # Grade pattern
        if 'normalized_grade' in cluster_data.columns:
            grade_counts = cluster_data['normalized_grade'].value_counts()
            if len(grade_counts) > 0:
                top_grade = grade_counts.index[0]
                if grade_counts.iloc[0] > len(cluster_data) * 0.5:
                    interpretations.append(f"Dominant {top_grade} grade")
                else:
                    interpretations.append(f"Mixed grades, mostly {top_grade}")
        
        if 'coating' in cluster_data.columns:
            coating_counts = cluster_data['coating'].value_counts()
            if len(coating_counts) > 0:
                top_coating = coating_counts.index[0]
                if coating_counts.iloc[0] > len(cluster_data) * 0.6:
                    interpretations.append(f"{top_coating} coated")
        
        if 'form' in cluster_data.columns:
            form_counts = cluster_data['form'].value_counts()
            if len(form_counts) > 0:
                top_form = form_counts.index[0]
                interpretations.append(f"{top_form} form")
        
        size_interpretation = self._interpret_size_pattern(cluster_data)
        if size_interpretation:
            interpretations.append(size_interpretation)
        
        if not interpretations:
            return "Diverse cluster with no clear dominant patterns"
        
        return "; ".join(interpretations)

    def _interpret_size_pattern(self, cluster_data: pd.DataFrame) -> str:
        """Interpret size patterns in cluster."""
        dimension_stats = []
        
        for dim in ['thickness', 'width', 'length']:
            min_col = f'{dim}_min'
            max_col = f'{dim}_max'
            
            if min_col in cluster_data.columns and max_col in cluster_data.columns:
                avg_size = (cluster_data[min_col].mean() + cluster_data[max_col].mean()) / 2
                if avg_size > 100:
                    size_cat = "large"
                elif avg_size > 10:
                    size_cat = "medium"
                else:
                    size_cat = "small"
                dimension_stats.append(f"{size_cat} {dim}")
        
        if dimension_stats:
            return f"Typically {' & '.join(dimension_stats)}"
        return ""

    def run_complete_analysis(self) -> Dict[str, pd.DataFrame]:
        """Run the complete analysis pipeline."""
        print("Starting complete RFQ analysis pipeline...")
        
        self.load_and_preprocess_data()
        
        self.engineer_features()
        
        print("\n" + "="*60)
        print("GENERATING TOP-3 SIMILARITY MATCHES (CSV OUTPUT)")
        print("="*60)
        top3_results = self.generate_top3_matches("top3.csv")
        
        ablation_results = self.run_ablation_analysis()
        
        metrics_results = self.run_alternative_metrics()
        
        clustering_results = self.run_clustering(8)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Generated files:")
        print("• top3.csv - Top-3 similarity matches (CSV file)")
        print("\nDisplayed in console:")
        print("• Ablation analysis - Feature group importance")
        print("• Alternative metrics - Comparison of similarity approaches") 
        print("• Clustering results - RFQ family groups and interpretations")
        print("• Cluster assignments - Which RFQs belong to which clusters")
        
        return {
            'top3_matches': top3_results,
            'ablation_analysis': ablation_results,
            'alternative_metrics': metrics_results,
            'clustering_results': clustering_results
        }


