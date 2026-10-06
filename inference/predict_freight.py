import joblib
import pandas as pd

from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict
        Input features for the freight cost prediction.

    Returns
    -------
    pd.DataFrame with predicted freight cost.
    """

    model = load_model()

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame(input_data)
    
    input_df["predicted_freight"] = model.predict(input_df).round()

    return input_df

if __name__== '__main__':
    #Example inference run(local testing)
    sample_data={
        "Dollars" :[18500, 9000]
    }
    prediction=predict_freight_cost(sample_data)
    print(prediction)
    