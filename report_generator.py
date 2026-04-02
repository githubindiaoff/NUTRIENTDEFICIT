from datetime import datetime

def generate_report(prediction, confidence):
    return f"""
    Nutrient Deficiency Analysis Report
    -------------------------------------
    Date: {datetime.now()}

    Predicted Deficiency:
    {prediction}

    Confidence Score:
    {confidence} %

    Model Used:
    BioBERT Fine-Tuned on Clinical Dataset
    """