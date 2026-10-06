from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "predict_flag_invoice.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]


def load_model(model_path=MODEL_PATH):
    """Load trained classifier model."""
    with open(model_path, "rb") as f:
        return joblib.load(f)


def predict_invoice_flag(input_data):
    """Predict invoice flag for new vendor invoices."""
    model = load_model()
    scaler = joblib.load(SCALER_PATH)

    input_df = pd.DataFrame(input_data)[FEATURES]
    scaled = scaler.transform(input_df)

    input_df["predicted_flag"] = model.predict(scaled).astype(int)
    return input_df


if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_item_quantity": [162],
        "total_item_dollars": [2476.0],
    }
    print(predict_invoice_flag(sample_data))