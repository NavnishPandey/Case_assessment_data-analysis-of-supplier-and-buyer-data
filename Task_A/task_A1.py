import pandas as pd
import numpy as np
import re
from datetime import datetime
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InventoryDataProcessor:
    """
    Complete pipeline for cleaning and joining supplier inventory datasets
    Addresses Task A.1 requirements: Clean & Normalize, Handle Missing Values, Join, Document Assumptions
    """

    def __init__(self):
        self.assumptions = []
        self.cleaning_stats = {}
        self.validation_rules = {
            'thickness_mm': {'min': 0.05, 'max': 100, 'description': 'Sheet metal thickness range'},
            'width_mm': {'min': 5, 'max': 5000, 'description': 'Sheet metal width range'},
            'weight_kg': {'min': 0.01, 'max': 10000, 'description': 'Realistic weight range'},
            'quantity': {'min': 0, 'max': 10000, 'description': 'Inventory quantity range'},
            'rp02': {'min': 50, 'max': 3000, 'description': 'Yield strength (MPa)'},
            'rm': {'min': 50, 'max': 3000, 'description': 'Tensile strength (MPa)'},
            'ag': {'min': 0, 'max': 100, 'description': 'Elongation percentage'},
            'ai': {'min': 0, 'max': 500, 'description': 'Impact energy (J)'}
        }

    def log_assumption(self, assumption: str):
        """Log data cleaning assumptions for documentation"""
        self.assumptions.append(f"{datetime.now().strftime('%H:%M:%S')} - {assumption}")
        logger.info(f"ASSUMPTION: {assumption}")

    def validate_numeric_column(self, df, col_name, remove_outliers=True):
        """Validate numeric columns against realistic ranges"""
        if col_name not in df.columns:
            return df

        original_count = df[col_name].notna().sum()

        # Convert to numeric
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

        # Apply validation rules if available
        if col_name in self.validation_rules and remove_outliers:
            rules = self.validation_rules[col_name]
            min_val, max_val = rules['min'], rules['max']

            # Count values outside range
            outliers = ((df[col_name] < min_val) | (df[col_name] > max_val)).sum()

            if outliers > 0:
                df.loc[(df[col_name] < min_val) | (df[col_name] > max_val), col_name] = np.nan
                self.log_assumption(f"Removed {outliers} outlier values in '{col_name}' outside range [{min_val}-{max_val}] - {rules['description']}")

        cleaned_count = df[col_name].notna().sum()
        conversion_lost = original_count - cleaned_count

        if conversion_lost > 0:
            self.log_assumption(f"Lost {conversion_lost} non-numeric values during conversion of '{col_name}'")

        return df

    def standardize_categorical(self, df, col_name, mapping_dict, default_value='unknown'):
        """Standardize categorical columns with mapping"""
        if col_name not in df.columns:
            return df

        original_unique = df[col_name].nunique()

        # Clean and standardize
        df[col_name] = df[col_name].astype(str).str.strip().str.lower()
        df[col_name] = df[col_name].replace({'nan': default_value, '': default_value, 'none': default_value})

        # Apply mapping
        df[col_name] = df[col_name].map(mapping_dict).fillna(df[col_name])

        final_unique = df[col_name].nunique()
        self.log_assumption(f"Standardized '{col_name}': {original_unique} → {final_unique} unique values")

        return df

    def clean_supplier_data1(self, df):
        """Clean and normalize supplier_data1 dataset"""
        logger.info("🔧 Cleaning Supplier Data 1...")
        df_clean = df.copy()

        # Step 1: Standardize column names
        column_mapping = {
            'Quality/Choice': 'quality_choice',
            'Grade': 'grade',
            'Finish': 'finish',
            'Thickness (mm)': 'thickness_mm',
            'Width (mm)': 'width_mm',
            'Description': 'description',
            'Gross weight (kg)': 'weight_kg',
            'Quantity': 'quantity',
            'RP02': 'rp02',
            'RM': 'rm',
            'AG': 'ag',
            'AI': 'ai'
        }

        # Handle various possible column name formats
        df_clean.columns = df_clean.columns.str.strip()
        for old_name, new_name in column_mapping.items():
            if old_name in df_clean.columns:
                df_clean.rename(columns={old_name: new_name}, inplace=True)

        self.log_assumption(f"Standardized column names for supplier 1: {list(column_mapping.values())}")

        # Step 2: Clean and validate numeric columns
        numeric_columns = ['thickness_mm', 'width_mm', 'weight_kg', 'quantity', 'rp02', 'rm', 'ag', 'ai']
        for col in numeric_columns:
            df_clean = self.validate_numeric_column(df_clean, col, remove_outliers=True)

        # Step 3: Standardize categorical columns

        # Quality/Choice standardization
        quality_mapping = {
            '1st': '1st', 'first': '1st', '1': '1st', 'i': '1st',
            '2nd': '2nd', 'second': '2nd', '2': '2nd', 'ii': '2nd',
            '3rd': '3rd', 'third': '3rd', '3': '3rd', 'iii': '3rd'
        }
        df_clean = self.standardize_categorical(df_clean, 'quality_choice', quality_mapping, 'unknown')

        # Grade standardization (keep uppercase for material grades)
        if 'grade' in df_clean.columns:
            df_clean['grade'] = df_clean['grade'].astype(str).str.strip().str.upper()
            df_clean['grade'] = df_clean['grade'].replace({'NAN': 'UNKNOWN', '': 'UNKNOWN'})

        # Finish standardization (German to English translation)
        finish_mapping = {
            'ungebeizt': 'unpickled',
            'gebeizt': 'pickled',
            'blank': 'bright',
            'geschliffen': 'ground',
            'geölt': 'oiled',
            'oiled': 'oiled'
        }
        df_clean = self.standardize_categorical(df_clean, 'finish', finish_mapping, 'unknown')

        # Step 4: Handle description field
        if 'description' in df_clean.columns:
            df_clean['description'] = df_clean['description'].astype(str).str.strip()
            df_clean['description'] = df_clean['description'].replace({'nan': 'No description', '': 'No description'})

            # Extract material type from description if grade is missing
            material_pattern = r'\b([A-Z]{1,4}\d{2,4}[A-Z]*)\b'
            extracted_materials = df_clean['description'].str.extract(material_pattern)[0]

            if 'grade' in df_clean.columns:
                grade_missing = df_clean['grade'].isin(['UNKNOWN', ''])
                df_clean.loc[grade_missing & extracted_materials.notna(), 'grade'] = extracted_materials
                material_filled = (grade_missing & extracted_materials.notna()).sum()
                if material_filled > 0:
                    self.log_assumption(f"Extracted {material_filled} material grades from description field")

        # Step 5: Add metadata
        df_clean['data_source'] = 'supplier_1'
        df_clean['material_type'] = df_clean.get('grade', 'UNKNOWN')
        df_clean['has_mechanical_data'] = df_clean[['rp02', 'rm', 'ag', 'ai']].notna().any(axis=1)

        # Step 6: Quality checks
        self.cleaning_stats['supplier_1'] = {
            'original_records': len(df),
            'cleaned_records': len(df_clean),
            'columns': list(df_clean.columns),
            'mechanical_data_available': df_clean['has_mechanical_data'].sum()
        }

        logger.info(f"✅ Supplier 1 cleaned: {len(df_clean)} records, {df_clean['has_mechanical_data'].sum()} with mechanical data")
        return df_clean

    def clean_supplier_data2(self, df):
        """Clean and normalize supplier_data2 dataset"""
        logger.info("🔧 Cleaning Supplier Data 2...")
        df_clean = df.copy()

        # Step 1: Standardize column names
        column_mapping = {
            'Material': 'material',
            'Description': 'description',
            'Article ID': 'article_id',
            'Weight (kg)': 'weight_kg',
            'Quantity': 'quantity',
            'Reserved': 'reserved'
        }

        df_clean.columns = df_clean.columns.str.strip()
        for old_name, new_name in column_mapping.items():
            if old_name in df_clean.columns:
                df_clean.rename(columns={old_name: new_name}, inplace=True)

        self.log_assumption(f"Standardized column names for supplier 2: {list(column_mapping.values())}")

        # Step 2: Parse material information (e.g., "DX51D +Z140")
        if 'material' in df_clean.columns:
            df_clean['material'] = df_clean['material'].astype(str).str.strip().str.upper()

            # Split material into base material and coating
            material_split = df_clean['material'].str.split('+', n=1, expand=True)
            df_clean['base_material'] = material_split[0].str.strip()

            if len(material_split.columns) > 1:
                df_clean['coating'] = material_split[1].str.strip()
            else:
                df_clean['coating'] = 'NONE'

            df_clean['coating'] = df_clean['coating'].fillna('NONE')
            df_clean['material_type'] = df_clean['base_material']
            df_clean['grade'] = df_clean['base_material']  # Use base material as grade

            coated_count = (df_clean['coating'] != 'NONE').sum()
            self.log_assumption(f"Parsed material field: {coated_count} records have coating information")

        # Step 3: Clean numeric columns
        numeric_columns = ['weight_kg', 'quantity']
        for col in numeric_columns:
            df_clean = self.validate_numeric_column(df_clean, col, remove_outliers=True)

        # Step 4: Standardize other fields

        # Description field
        if 'description' in df_clean.columns:
            df_clean['description'] = df_clean['description'].astype(str).str.strip()
            df_clean['description'] = df_clean['description'].replace({'nan': 'No description', '': 'No description'})

            # Extract finish information from description
            df_clean['finish'] = df_clean['description'].apply(self._extract_finish_from_description)

        # Article ID standardization
        if 'article_id' in df_clean.columns:
            df_clean['article_id'] = df_clean['article_id'].astype(str).str.strip()
            df_clean['article_id'] = df_clean['article_id'].replace({'nan': None, '': None})

        # Reserved status (boolean conversion)
        if 'reserved' in df_clean.columns:
            df_clean['reserved'] = df_clean['reserved'].astype(str).str.strip().str.upper()
            reserved_mapping = {'YES': True, 'Y': True, 'TRUE': True, '1': True, 'RESERVED': True, 'X': True}
            df_clean['reserved'] = df_clean['reserved'].map(reserved_mapping).fillna(False)
            reserved_count = df_clean['reserved'].sum()
            self.log_assumption(f"Converted reserved status: {reserved_count} items are reserved")

        # Step 5: Add metadata and missing columns for schema alignment
        df_clean['data_source'] = 'supplier_2'
        df_clean['quality_choice'] = 'unknown'  # Supplier 2 doesn't specify quality
        df_clean['thickness_mm'] = np.nan  # Supplier 2 doesn't provide dimensions
        df_clean['width_mm'] = np.nan
        df_clean['has_mechanical_data'] = False  # Supplier 2 doesn't provide mechanical data

        # Initialize mechanical properties columns with NaN
        for prop in ['rp02', 'rm', 'ag', 'ai']:
            df_clean[prop] = np.nan

        # Step 6: Quality checks
        self.cleaning_stats['supplier_2'] = {
            'original_records': len(df),
            'cleaned_records': len(df_clean),
            'columns': list(df_clean.columns),
            'reserved_items': df_clean['reserved'].sum(),
            'unique_materials': df_clean['material_type'].nunique()
        }

        logger.info(f"✅ Supplier 2 cleaned: {len(df_clean)} records, {df_clean['reserved'].sum()} reserved items")
        return df_clean

    def _extract_finish_from_description(self, description):
        """Extract finish information from description text"""
        if pd.isna(description) or description == '' or description == 'No description':
            return 'unknown'

        desc_lower = str(description).lower()

        finish_keywords = {
            'oil': 'oiled',
            'oiled': 'oiled',
            'geölt': 'oiled',
            'pick': 'pickled',
            'gebeizt': 'pickled',
            'pickled': 'pickled',
            'bright': 'bright',
            'blank': 'bright',
            'ground': 'ground',
            'geschliffen': 'ground',
            'coated': 'coated',
            'galvanized': 'galvanized',
            'galv': 'galvanized'
        }

        for keyword, finish in finish_keywords.items():
            if keyword in desc_lower:
                return finish

        return 'unknown'

    def handle_missing_values(self, df):
        """Comprehensive missing value handling with documented strategies"""
        logger.info("🔄 Handling missing values...")
        df_clean = df.copy()

        missing_before = df_clean.isnull().sum()
        total_missing_before = missing_before.sum()

        if total_missing_before == 0:
            logger.info("✅ No missing values found!")
            return df_clean

        logger.info(f"Found {total_missing_before} missing values across {(missing_before > 0).sum()} columns")

        # Define strategy for each column
        for col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count == 0:
                continue

            missing_pct = (missing_count / len(df_clean)) * 100
            logger.info(f"  {col}: {missing_count} missing ({missing_pct:.1f}%)")

            # Apply column-specific strategy
            if col in ['thickness_mm', 'width_mm']:
                # Use median for dimensional data (robust to outliers)
                median_val = df_clean[col].median()
                if pd.notna(median_val):
                    df_clean[col] = df_clean[col].fillna(median_val)
                    self.log_assumption(f"Filled {missing_count} missing '{col}' with median: {median_val:.2f}mm")
                else:
                    # If all values are missing, use industry defaults
                    default_val = 1.0 if col == 'thickness_mm' else 1000.0
                    df_clean[col] = df_clean[col].fillna(default_val)
                    self.log_assumption(f"All '{col}' values missing, used industry default: {default_val}mm")

            elif col == 'weight_kg':
                # Use median for weight
                median_val = df_clean[col].median()
                if pd.notna(median_val):
                    df_clean[col] = df_clean[col].fillna(median_val)
                    self.log_assumption(f"Filled {missing_count} missing weights with median: {median_val:.2f}kg")
                else:
                    df_clean[col] = df_clean[col].fillna(1.0)
                    self.log_assumption(f"All weights missing, used default: 1.0kg")

            elif col == 'quantity':
                # Quantity defaults to 1 (assume at least one item exists)
                df_clean[col] = df_clean[col].fillna(1)
                self.log_assumption(f"Filled {missing_count} missing quantities with 1 (minimum inventory)")

            elif col in ['quality_choice', 'finish']:
                # Use mode for categorical data, fallback to 'unknown'
                mode_values = df_clean[col].mode()
                if len(mode_values) > 0 and pd.notna(mode_values.iloc[0]):
                    mode_val = mode_values.iloc[0]
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    self.log_assumption(f"Filled {missing_count} missing '{col}' with mode: '{mode_val}'")
                else:
                    df_clean[col] = df_clean[col].fillna('unknown')
                    self.log_assumption(f"Filled {missing_count} missing '{col}' with 'unknown'")

            elif col in ['grade', 'material_type']:
                # Try to use the other field if available, otherwise 'UNKNOWN'
                if col == 'grade' and 'material_type' in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(df_clean['material_type'])
                elif col == 'material_type' and 'grade' in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(df_clean['grade'])

                # Fill remaining with 'UNKNOWN'
                remaining_missing = df_clean[col].isnull().sum()
                df_clean[col] = df_clean[col].fillna('UNKNOWN')
                self.log_assumption(f"Filled {missing_count} missing '{col}' (cross-filled or 'UNKNOWN')")

            elif col == 'reserved':
                # Reserved defaults to False (not reserved)
                df_clean[col] = df_clean[col].fillna(False)
                self.log_assumption(f"Filled {missing_count} missing reservation status with False")

            elif col == 'article_id':
                # Generate unique article IDs for missing values
                missing_mask = df_clean[col].isnull()
                unique_ids = [f"AUTO_ID_{i:06d}" for i in range(missing_mask.sum())]
                df_clean.loc[missing_mask, col] = unique_ids
                self.log_assumption(f"Generated {missing_count} automatic article IDs")

            elif col in ['rp02', 'rm', 'ag', 'ai']:
                # Mechanical properties: don't fill, leave as NaN (not all materials have this data)
                self.log_assumption(f"Left {missing_count} mechanical property '{col}' as NaN (supplier-specific data)")
                continue

            else:
                # Default strategy based on data type
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].fillna('unknown')
                    self.log_assumption(f"Filled {missing_count} missing '{col}' with 'unknown'")
                elif df_clean[col].dtype == 'bool':
                    df_clean[col] = df_clean[col].fillna(False)
                    self.log_assumption(f"Filled {missing_count} missing '{col}' with False")
                else:
                    df_clean[col] = df_clean[col].fillna(0)
                    self.log_assumption(f"Filled {missing_count} missing '{col}' with 0")

        # Final check
        missing_after = df_clean.isnull().sum().sum()
        logger.info(f"✅ Missing values: {total_missing_before} → {missing_after}")

        return df_clean

    def join_datasets(self, df1_clean, df2_clean):
        """Join cleaned datasets into unified inventory schema"""
        logger.info("🔗 Creating unified inventory dataset...")

        # Define target schema
        target_columns = [
            'article_id', 'data_source', 'material_type', 'grade', 'quality_choice',
            'finish', 'thickness_mm', 'width_mm', 'weight_kg', 'quantity',
            'reserved', 'description', 'coating', 'has_mechanical_data',
            'rp02', 'rm', 'ag', 'ai'
        ]

        # Ensure all target columns exist in both datasets
        for df in [df1_clean, df2_clean]:
            for col in target_columns:
                if col not in df.columns:
                    if col in ['coating']:
                        df[col] = 'UNKNOWN'
                    elif col in ['reserved']:
                        df[col] = False
                    elif col in ['has_mechanical_data']:
                        df[col] = False
                    elif col in ['rp02', 'rm', 'ag', 'ai', 'thickness_mm', 'width_mm']:
                        df[col] = np.nan
                    else:
                        df[col] = 'unknown'

        # Generate unique article IDs where missing
        for i, df in enumerate([df1_clean, df2_clean], 1):
            missing_ids = df['article_id'].isnull()
            if missing_ids.any():
                prefix = f"SUP{i}"
                df.loc[missing_ids, 'article_id'] = [f"{prefix}_{j:06d}" for j in range(missing_ids.sum())]

        # Combine datasets
        inventory_df = pd.concat([df1_clean[target_columns], df2_clean[target_columns]],
                                ignore_index=True, sort=False)

        # Add calculated fields
        inventory_df['total_weight_kg'] = inventory_df['weight_kg'] * inventory_df['quantity']
        inventory_df['available_quantity'] = inventory_df['quantity'] - inventory_df['reserved'].astype(int)
        inventory_df['processing_date'] = datetime.now().strftime('%Y-%m-%d')
        inventory_df['processing_time'] = datetime.now().strftime('%H:%M:%S')

        # Final data validation
        inventory_df = self._final_validation(inventory_df)

        self.log_assumption(f"Unified dataset created with {len(inventory_df)} total records")
        self.log_assumption(f"Schema: {len(inventory_df.columns)} columns with standardized naming")

        logger.info(f"✅ Unified inventory: {len(inventory_df)} records from {inventory_df['data_source'].nunique()} suppliers")
        return inventory_df

    def _final_validation(self, df):
        """Final validation and cleanup of unified dataset"""

        # Ensure quantity is non-negative integer
        df['quantity'] = df['quantity'].clip(lower=0).astype(int)
        df['available_quantity'] = df['available_quantity'].clip(lower=0).astype(int)

        # Ensure total_weight_kg is non-negative
        df['total_weight_kg'] = df['total_weight_kg'].clip(lower=0)

        # Standardize text fields
        text_columns = ['material_type', 'grade', 'quality_choice', 'finish', 'description']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Remove any duplicate records based on key fields
        key_fields = ['article_id', 'data_source']
        duplicates = df.duplicated(subset=key_fields, keep='first')
        if duplicates.any():
            duplicate_count = duplicates.sum()
            df = df[~duplicates]
            self.log_assumption(f"Removed {duplicate_count} duplicate records based on article_id + data_source")

        return df

    def generate_comprehensive_report(self, inventory_df):
        """Generate detailed cleaning and processing report"""

        report = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'total_records': len(inventory_df),
                'total_columns': len(inventory_df.columns),
                'data_sources': inventory_df['data_source'].value_counts().to_dict()
            },
            'data_quality': {
                'completeness_pct': round(((inventory_df.notna().sum().sum()) /
                                         (inventory_df.shape[0] * inventory_df.shape[1])) * 100, 2),
                'records_with_mechanical_data': int(inventory_df['has_mechanical_data'].sum()),
                'records_with_dimensions': int(inventory_df[['thickness_mm', 'width_mm']].notna().any(axis=1).sum()),
                'reserved_items': int(inventory_df['reserved'].sum()),
                'total_inventory_weight_kg': round(inventory_df['total_weight_kg'].sum(), 2),
                'unique_materials': int(inventory_df['material_type'].nunique()),
                'unique_grades': int(inventory_df['grade'].nunique())
            },
            'missing_data': inventory_df.isnull().sum().to_dict(),
            'column_info': {col: str(dtype) for col, dtype in inventory_df.dtypes.items()},
            'assumptions_made': self.assumptions,
            'cleaning_stats': self.cleaning_stats
        }

        return report

    def save_results(self, inventory_df, report, output_file='inventory_dataset.csv'):
        """Save the cleaned dataset and comprehensive report"""

        # Save main dataset
        inventory_df.to_csv(output_file, index=False)
        logger.info(f"💾 Inventory dataset saved: {output_file}")

        # Save detailed report
        report_file = output_file.replace('.csv', '_cleaning_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("INVENTORY DATA CLEANING & JOINING REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Processing info
            f.write(f"Processing Date: {report['processing_info']['timestamp']}\n")
            f.write(f"Total Records: {report['processing_info']['total_records']:,}\n")
            f.write(f"Total Columns: {report['processing_info']['total_columns']}\n\n")

            # Data sources
            f.write("DATA SOURCES:\n")
            for source, count in report['processing_info']['data_sources'].items():
                f.write(f"  • {source}: {count:,} records\n")
            f.write("\n")

            # Data quality metrics
            f.write("DATA QUALITY METRICS:\n")
            dq = report['data_quality']
            f.write(f"  • Data Completeness: {dq['completeness_pct']}%\n")
            f.write(f"  • Records with Mechanical Data: {dq['records_with_mechanical_data']:,}\n")
            f.write(f"  • Records with Dimensions: {dq['records_with_dimensions']:,}\n")
            f.write(f"  • Reserved Items: {dq['reserved_items']:,}\n")
            f.write(f"  • Total Inventory Weight: {dq['total_inventory_weight_kg']:,.2f} kg\n")
            f.write(f"  • Unique Materials: {dq['unique_materials']:,}\n")
            f.write(f"  • Unique Grades: {dq['unique_grades']:,}\n\n")

            # Missing data summary
            f.write("REMAINING MISSING DATA:\n")
            missing_cols = {k: v for k, v in report['missing_data'].items() if v > 0}
            if missing_cols:
                for col, count in missing_cols.items():
                    pct = (count / report['processing_info']['total_records']) * 100
                    f.write(f"  • {col}: {count:,} ({pct:.1f}%)\n")
            else:
                f.write("  ✅ No missing data in final dataset!\n")
            f.write("\n")

            # Assumptions documentation
            f.write("DATA CLEANING ASSUMPTIONS:\n")
            for i, assumption in enumerate(report['assumptions_made'], 1):
                f.write(f"{i:2d}. {assumption}\n")
            f.write("\n")

            # Column information
            f.write("COLUMN DATA TYPES:\n")
            for col, dtype in report['column_info'].items():
                f.write(f"  • {col}: {dtype}\n")

        logger.info(f"📋 Detailed report saved: {report_file}")

        return output_file, report_file

    def process_datasets(self, file1_path, file2_path, output_file='inventory_dataset.csv'):
        """
        Main processing pipeline for Task A.1 - Clean & Join

        Steps:
        1. Load datasets
        2. Clean and normalize each dataset
        3. Handle missing/inconsistent values
        4. Join into unified schema
        5. Generate final dataset
        6. Document all assumptions
        """

        start_time = datetime.now()
        logger.info("🚀 Starting Inventory Data Processing Pipeline")
        logger.info("=" * 60)

        try:
            # Step 1: Load datasets
            logger.info("📁 Step 1: Loading datasets...")

            if not Path(file1_path).exists():
                raise FileNotFoundError(f"Supplier data 1 not found: {file1_path}")
            if not Path(file2_path).exists():
                raise FileNotFoundError(f"Supplier data 2 not found: {file2_path}")

            # Load based on file extension
            if file1_path.endswith('.xlsx'):
                df1 = pd.read_excel(file1_path)
            else:
                df1 = pd.read_csv(file1_path)

            if file2_path.endswith('.xlsx'):
                df2 = pd.read_excel(file2_path)
            else:
                df2 = pd.read_csv(file2_path)

            logger.info(f"  ✅ Loaded {file1_path}: {len(df1)} records, {len(df1.columns)} columns")
            logger.info(f"  ✅ Loaded {file2_path}: {len(df2)} records, {len(df2.columns)} columns")

            self.log_assumption(f"Loaded supplier_data1: {len(df1)} records from {file1_path}")
            self.log_assumption(f"Loaded supplier_data2: {len(df2)} records from {file2_path}")

            # Step 2: Clean and normalize each dataset
            logger.info("\n🔧 Step 2: Cleaning and normalizing datasets...")

            df1_clean = self.clean_supplier_data1(df1)
            df2_clean = self.clean_supplier_data2(df2)

            # Step 3: Handle missing/inconsistent values
            logger.info("\n🔄 Step 3: Handling missing and inconsistent values...")

            df1_clean = self.handle_missing_values(df1_clean)
            df2_clean = self.handle_missing_values(df2_clean)

            # Step 4: Join into unified schema
            logger.info("\n🔗 Step 4: Creating unified inventory dataset...")

            inventory_dataset = self.join_datasets(df1_clean, df2_clean)

            # Step 5: Final data quality check
            logger.info("\n✅ Step 5: Final validation and quality check...")

            # Check for any remaining null values
            remaining_nulls = inventory_dataset.isnull().sum()
            total_nulls = remaining_nulls.sum()

            if total_nulls > 0:
                logger.warning(f"⚠️  {total_nulls} null values remain - force cleaning...")
                # Force clean any remaining nulls
                for col in inventory_dataset.columns:
                    if remaining_nulls[col] > 0:
                        if inventory_dataset[col].dtype == 'object':
                            inventory_dataset[col] = inventory_dataset[col].fillna('unknown')
                        elif inventory_dataset[col].dtype in ['int64', 'float64']:
                            if col in ['rp02', 'rm', 'ag', 'ai', 'thickness_mm', 'width_mm']:
                                # Leave mechanical properties and dimensions as NaN (valid for missing data)
                                continue
                            else:
                                inventory_dataset[col] = inventory_dataset[col].fillna(0)
                        else:
                            inventory_dataset[col] = inventory_dataset[col].fillna(False)

                self.log_assumption("Final cleanup: Force-filled remaining null values with appropriate defaults")

            # Step 6: Generate comprehensive report
            logger.info("\n📋 Step 6: Generating comprehensive documentation...")

            report = self.generate_comprehensive_report(inventory_dataset)

            # Step 7: Save results
            logger.info("\n💾 Step 7: Saving results...")

            output_files = self.save_results(inventory_dataset, report, output_file)

            # Processing summary
            end_time = datetime.now()
            processing_time = end_time - start_time

            logger.info("\n" + "=" * 60)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"⏱️  Processing Time: {processing_time}")
            logger.info(f"📊 Total Records: {len(inventory_dataset):,}")
            logger.info(f"📈 Data Completeness: {report['data_quality']['completeness_pct']}%")
            logger.info(f"🏭 Suppliers: {len(report['processing_info']['data_sources'])}")
            logger.info(f"📋 Assumptions Made: {len(self.assumptions)}")
            logger.info(f"💾 Files Generated: {len(output_files)}")

            for file_type, file_path in zip(['Dataset', 'Report', 'Sample'], output_files):
                logger.info(f"    • {file_type}: {file_path}")

            # Final validation message
            final_nulls = inventory_dataset.isnull().sum().sum()
            if final_nulls == 0:
                logger.info("✅ SUCCESS: Zero null values in final dataset!")
            else:
                logger.info(f"ℹ️  INFO: {final_nulls} null values remain (valid missing data)")

            return inventory_dataset, report

        except FileNotFoundError as e:
            logger.error(f"❌ File Error: {str(e)}")
            logger.error("💡 Please ensure both supplier data files exist in the specified location")
            raise

        except Exception as e:
            logger.error(f"❌ Processing Error: {str(e)}")
            logger.error("💡 Check input data format and try again")
            raise

        finally:
            # Always log final assumptions summary
            if self.assumptions:
                logger.info(f"\n📝 TOTAL ASSUMPTIONS DOCUMENTED: {len(self.assumptions)}")


# Utility function for easy execution
def clean_and_join_inventory(supplier1_file, supplier2_file, output_file='inventory_dataset.csv'):
    """
    Convenience function to run the complete inventory cleaning pipeline

    Args:
        supplier1_file (str): Path to supplier 1 data file (Excel or CSV)
        supplier2_file (str): Path to supplier 2 data file (Excel or CSV)
        output_file (str): Output filename for cleaned inventory dataset

    Returns:
        tuple: (cleaned_dataset, processing_report)
    """
    processor = InventoryDataProcessor()
    return processor.process_datasets(supplier1_file, supplier2_file, output_file)


