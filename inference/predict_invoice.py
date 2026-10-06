import joblib
import pandas as pd

from predict_freight import predict_freight_cost

MODEL_PATH = "models/predict_flag_invoice.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict
        Input features for the invoice prediction.

    Returns
    -------
    pd.DataFrame with predicted flag.
    """

    model = load_model()

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame(input_data)

    input_df["predicted_flag"] = model.predict(input_df).round()

    return input_df

if __name__== '__main__':
    #Example inference run(local testing)
    sample_data={
        "Dollars" :[18500, 9000]
    }
    prediction=predict_freight_cost(sample_data)
    print(prediction)
    