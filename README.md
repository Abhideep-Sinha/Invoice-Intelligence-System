# Inventory & Invoice Analytics

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abhideep-sinha-invoice-intelligence-system-app-zctqvf.streamlit.app/)

**🚀 Live demo: https://abhideep-sinha-invoice-intelligence-system-app-zctqvf.streamlit.app/**

A machine learning project built on a vendor inventory database. It includes two models and a Streamlit web app to use them:

1. **Freight Cost Prediction**: predicts the freight (shipping) cost for an order.
2. **Invoice Risk Flagging**: classifies vendor invoices as normal or risky (needs manual review).

---

## Table of Contents

- [Live Demo](#live-demo)
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Data](#data)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit App](#streamlit-app)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

---

## Live Demo

Try the deployed app here: **[Vendor Invoice Intelligence Portal](https://abhideep-sinha-invoice-intelligence-system-app-zctqvf.streamlit.app/)**

It has two modules, chosen from the sidebar: **Freight Cost Prediction** and **Invoice Manual Approval Flag**.

> The app is hosted on Streamlit Community Cloud. If it has been idle, it may go to sleep. Click "Yes, get this app back up!" and wait a minute for it to restart.

---

## Problem Statement

Procurement teams handle thousands of vendor invoices. Two common problems:

- **Freight costs are hard to estimate** before an invoice arrives, which makes budgeting difficult.
- **Invoice errors and delays** (an invoice total that doesn't match what was received, or very late deliveries) are found manually, which is slow.

This project addresses both with simple, explainable models.

---

## Project Structure

```
intelligent-voice/
├── data/
│   └── inventory.db                 # SQLite database (purchases, vendor_invoice, ...)
├── freight_cost_prediction/
│   ├── data_preprocessing.py        # Load data, split into train/test
│   ├── model_evaluation.py          # Train and evaluate the freight model
│   └── train.py                     # Entry point: trains and saves the model
├── invoice_flagging/
│   ├── data_preprocessing.py        # Load data, create labels, split, scale
│   ├── modelling_evaluation.py      # Random Forest + GridSearchCV, evaluation
│   └── train.py                     # Entry point: trains and saves the model
├── models/
│   ├── predict_freight_model.pkl    # Trained freight model
│   ├── predict_flag_invoice.pkl     # Trained invoice flagging model
│   └── scaler.pkl                   # StandardScaler for invoice features
├── notebook/                        # Exploration and analysis notebooks
├── inference/
│   ├── predict_freight.py           # Freight prediction helper used by the app
│   └── predict_invoice.py           # Invoice flag prediction helper used by the app
├── app.py                           # Streamlit UI
└── README.md
```

---

## Data

The data comes from a SQLite database, `data/inventory.db`. The code uses these tables:

| Table | Used for |
|---|---|
| `purchases` | Purchase order lines: PO number, brand, quantity, dollars, PO date, receiving date |
| `vendor_invoice` | Invoice details: quantity, dollars, freight, invoice date, PO date |

---

## How It Works

### 1. Freight Cost Prediction

- **Goal:** predict the freight cost of an order.
- **Model:** a scikit-learn linear regression model.
- **Input:** `Quantity` and invoice `Dollars`.
- **Output:** predicted freight cost, rounded.
- **Saved to:** `models/predict_freight_model.pkl`

### 2. Invoice Risk Flagging

**Step 1: Data loading.** A SQL query joins invoice data with per-PO aggregates from purchases:

- `total_brands`: number of distinct brands on the PO
- `total_item_quantity`, `total_item_dollars`: totals from the purchase lines
- `avg_receiving_delay`: average days between PO date and receiving date
- `invoice_quantity`, `invoice_dollars`, `Freight`, `days_po_to_invoice`: from the invoice

**Step 2: Labelling.** An invoice is flagged as risky (`flag_invoice = 1`) if either rule holds:

- the invoice amount differs from the total item amount by more than **$5**, or
- the average receiving delay is more than **10 days**.

Otherwise it is `0`.

**Step 3: Features and target.**

| Features | Target |
|---|---|
| `invoice_quantity`, `invoice_dollars`, `Freight`, `total_item_quantity`, `total_item_dollars` | `flag_invoice` |

**Step 4: Preprocessing.** 80/20 train/test split (`random_state=42`), then `StandardScaler` fitted on the training set only. The scaler is saved to `models/scaler.pkl`.

**Step 5: Model training.** A `RandomForestClassifier` tuned with `GridSearchCV` (5-fold cross-validation, scored by F1):

| Parameter | Values tried |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | None, 4, 5, 6 |
| `min_samples_split` | 2, 3, 5 |
| `min_samples_leaf` | 1, 2, 5 |
| `criterion` | gini, entropy |

That is 216 combinations, or 1,080 fits across the folds.

**Step 6: Evaluation and saving.** Accuracy and a classification report (precision, recall, F1) are printed on the test set. The best model is saved to `models/predict_flag_invoice.pkl`.

> **Note:** the risk label is built from rules on `invoice_dollars`, `total_item_dollars` and `avg_receiving_delay`. The model then learns from related columns, so high accuracy partly reflects that the labels are rule-based. The project is best read as a demonstration of the full ML pipeline.

---

## Installation

**Requirements:** Python 3.9 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/Abhideep-Sinha/Invoice-Intelligence-System.git
cd Invoice-Intelligence-System

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy scikit-learn joblib streamlit plotly
```

Or create a `requirements.txt`:

```
pandas
numpy
scikit-learn
joblib
streamlit
plotly
```

and run `pip install -r requirements.txt`.

`data/inventory.db` is about 405 MB, which is over GitHub's 100 MB file limit, so it is **not included in the repository**. To retrain the models, download it separately and place it in the `data/` folder. (TODO: add the download link.) The pre-trained models in `models/` are enough to run the app.

---

## Usage

### Train the freight model

```bash
cd freight_cost_prediction
python train.py
```

### Train the invoice flagging model

```bash
cd invoice_flagging
python train.py
```

This takes a few minutes because of the grid search, and prints results when it finishes. Both models are saved in the shared `models/` folder.

### Run a prediction from the command line

From the project root, call the inference functions directly:

```python
from inference.predict_freight import predict_freight_cost

predict_freight_cost({"Quantity": [1200], "Dollars": [18500.0]})
```

---

## Streamlit App

Use the hosted version at https://abhideep-sinha-invoice-intelligence-system-app-zctqvf.streamlit.app/, or run it locally. Start the web interface from the project root:

```bash
streamlit run app.py
```

The portal is titled **Vendor Invoice Intelligence Portal**. Use the sidebar to choose a module:

**1. Freight Cost Prediction**

| Input | Description |
|---|---|
| Quantity | Number of items on the invoice |
| Invoice Dollars | Invoice amount |

Click **Predict Freight Cost** to see the estimated freight cost.

**2. Invoice Manual Approval Flag**

| Input | Description |
|---|---|
| Invoice Quantity | Quantity on the invoice |
| Invoice Dollars | Invoice amount |
| Freight Cost | Freight charged |
| Total Item Quantity | Quantity across the PO's items |
| Total Item Dollars | Dollar total across the PO's items |

Click **Evaluate Invoice Risk**. The app shows either **MANUAL APPROVAL required** or **Safe for auto-approval**.

(TODO: add screenshots, for example `![App screenshot](docs/screenshot.png)`)

---

## Tech Stack

- **Python**
- **pandas**: data handling
- **SQLite**: data storage
- **scikit-learn**: models, scaling, grid search, metrics
- **joblib**: saving and loading models
- **Streamlit**: user interface
- **Plotly**: charts (imported in the app)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `sqlite3.OperationalError: unable to open database file` | The database path is wrong. Use a path relative to the script, for example `Path(__file__).resolve().parent.parent / "data" / "inventory.db"`. |
| `FileNotFoundError: models/scaler.pkl` | The `models/` folder does not exist where you ran the script. Create it, or build paths from `Path(__file__)`. |
| Models saved in the wrong `models/` folder | Relative paths depend on the folder you run from. Use `Path(__file__)`-based paths. |
| `ValueError: setting an array element with a sequence` | Input to `model.predict` contains lists in cells. Pass plain numbers when building the DataFrame. |
| `ModuleNotFoundError: predict_freight` in the app | Inside `inference/`, import with the full path: `from inference.predict_freight import predict_freight_cost`. |
| `ModuleNotFoundError: modeling_evaluation` | The import name must match the file name (`modelling_evaluation.py`). |

---

## Future Improvements

- Add more features for freight prediction (distance, vendor, weight)
- Compare more models (Gradient Boosting, XGBoost) for both tasks
- Label invoices with real review outcomes instead of fixed rules
- Add unit tests
- Add a Docker setup for self-hosting

---

## Author

Abhideep Sinha