from Task_A.task_A1 import clean_and_join_inventory
from Task_B.task_b import RFQSimilarityAnalyzer

SUPPLIER1_FILE = r"C:\Users\Alka\Documents\Case_assessment_data-analysis-of-supplier-and-buyer-data\taskA_data\supplier_data1.xlsx"
SUPPLIER2_FILE = r"C:\Users\Alka\Documents\Case_assessment_data-analysis-of-supplier-and-buyer-data\taskA_data\supplier_data2.xlsx"
OUTPUT_FILE = "inventory_dataset.csv"

RFQ_FILE = r"C:\Users\Alka\Documents\Case_assessment_data-analysis-of-supplier-and-buyer-data\taskB_data\rfq.csv"
REFERENCE_FILE = r"C:\Users\Alka\Documents\Case_assessment_data-analysis-of-supplier-and-buyer-data\taskB_data\reference_properties.tsv"


def run_task_a():
    """Runs Task A.1 - Clean and join supplier inventory datasets."""
    try:
        # Run pipeline
        cleaned_dataset, report = clean_and_join_inventory(
            SUPPLIER1_FILE,
            SUPPLIER2_FILE,
            OUTPUT_FILE
        )

        # Print key metrics
        print("\n📋 Key Metrics:")
        dq = report["data_quality"]
        print(f"   • Records with Mechanical Data: {dq['records_with_mechanical_data']:,}")
        print(f"   • Records with Dimensions: {dq['records_with_dimensions']:,}")
        print(f"   • Total Inventory Weight: {dq['total_inventory_weight_kg']:,.2f} kg")
        print(f"   • Unique Materials: {dq['unique_materials']:,}")
        print(f"   • Reserved Items: {dq['reserved_items']:,}")

        print("\n✅ Task A.1 Complete - Inventory dataset successfully created!")
        print("📋 All assumptions documented in the cleaning report")

    except FileNotFoundError:
        print("\n❌ ERROR: Input files not found!")
        print("💡 Please ensure the following files exist:")
        print(f"   • {SUPPLIER1_FILE}")
        print(f"   • {SUPPLIER2_FILE}")
        print("\n📝 Expected file format:")
        print("   • Excel (.xlsx) or CSV (.csv)")
        print("   • Headers as described in task documentation")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("💡 Please check your input data and try again")

    finally:
        print("\n📖 For detailed information, check the generated cleaning report.")
        print("🔍 Review assumptions and data quality metrics before proceeding to next tasks.")


def run_task_b():
    """Runs Task B - RFQ Similarity Analysis."""
    try:
        analyzer = RFQSimilarityAnalyzer(
            rfq_file=RFQ_FILE,
            reference_file=REFERENCE_FILE,
        )
        analyzer.run_complete_analysis()

        print("\n✅ Task B Complete - RFQ similarity analysis finished!")

    except FileNotFoundError:
        print("\n❌ ERROR: RFQ or Reference file not found!")
        print("💡 Please ensure the following files exist:")
        print(f"   • {RFQ_FILE}")
        print(f"   • {REFERENCE_FILE}")

    except Exception as e:
        print(f"\n❌ ERROR during Task B: {str(e)}")
        print("💡 Please check your input data and try again")


if __name__ == "__main__":
    print("\n🚀 Starting Task A.1 - Inventory Processing...")
    run_task_a()

    print("\n🚀 Starting Task B - RFQ Similarity Analysis...")
    run_task_b()
