from .utils.task_B1 import task_b1_pipline
from .utils.task_B2 import task_b2_pipline   
from .utils.task_B3 import task_b3_pipline



def RFQSimilarityAnalyzer(rfq_path, reference_path):
    
    task_b1_pipline(
        rfq_path=rfq_path,
        reference_path=reference_path,
    )
    task_b2_pipline(
        rfq_path=rfq_path,
        reference_path=reference_path,
    )
    task_b3_pipline(
        rfq_path=rfq_path,
        reference_path=reference_path,
    )
    print("\n✅ RFQ Similarity Analysis Complete - Results saved to 'top3.csv'")
    

