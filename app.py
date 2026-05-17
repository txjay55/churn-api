# ============================================================
#  ChurnGuard AI — FastAPI Backend
#  Models: LightGBM + DNN Ensemble (+ XGBoost loaded for comparison)
#  Matches metadata.json exactly
# ============================================================
#  Run: uvicorn app:app --reload --port 8000
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import json
import os
import lightgbm as lgb
from tensorflow import keras

# ── Load models & metadata ───────────────────────────────────
SAVE_DIR = "saved_models"

print("⏳ Loading models...")

lgb_model  = lgb.Booster(model_file=f"{SAVE_DIR}/lgb_model.txt")
dnn_model  = keras.models.load_model(f"{SAVE_DIR}/dnn_model.keras")
scaler     = joblib.load(f"{SAVE_DIR}/scaler.pkl")

# Load SHAP explainer if available
try:
    explainer = joblib.load(f"{SAVE_DIR}/lgb_shap_explainer.pkl")
    SHAP_READY = True
except:
    import shap
    explainer = shap.TreeExplainer(lgb_model)
    SHAP_READY = False
    print("⚠️  SHAP explainer rebuilt from lgb_model (no pkl found)")

with open(f"{SAVE_DIR}/metadata.json") as f:
    meta = json.load(f)

# ── From your exact metadata.json ────────────────────────────
FEATURE_NAMES = meta["feature_names"]   # 29 features
NUM_COLS      = meta["num_cols"]        # ["tenure","MonthlyCharges","TotalCharges","SeniorCitizen","ChargesPerMonth"]
THRESHOLD     = meta["best_threshold"]  # 0.52
LGB_W         = meta["lgb_weight"]     # 0.55
DNN_W         = meta["dnn_weight"]     # 0.45

print("✅ All models loaded!")
print(f"   Features  : {len(FEATURE_NAMES)}")
print(f"   Threshold : {THRESHOLD:.3f}")
print(f"   Weights   : LGB={LGB_W} · DNN={DNN_W}")

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Dual-model churn prediction: LightGBM + DNN Ensemble",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema — matches your 29 feature_names exactly ───
class CustomerInput(BaseModel):
    # Basic
    gender: int              = Field(0, ge=0, le=1,   description="0=Female 1=Male")
    SeniorCitizen: int       = Field(0, ge=0, le=1)
    Partner: int             = Field(0, ge=0, le=1)
    Dependents: int          = Field(0, ge=0, le=1)
    tenure: float            = Field(1, ge=0, le=72)
    # Services
    PhoneService: int        = Field(1, ge=0, le=1)
    MultipleLines: int       = Field(0, ge=0, le=1)
    OnlineSecurity: int      = Field(0, ge=0, le=1)
    OnlineBackup: int        = Field(0, ge=0, le=1)
    DeviceProtection: int    = Field(0, ge=0, le=1)
    TechSupport: int         = Field(0, ge=0, le=1)
    StreamingTV: int         = Field(0, ge=0, le=1)
    StreamingMovies: int     = Field(0, ge=0, le=1)
    # Billing
    PaperlessBilling: int    = Field(0, ge=0, le=1)
    MonthlyCharges: float    = Field(29.85, ge=0, le=200)
    TotalCharges: float      = Field(29.85, ge=0)
    # Internet service (one-hot — pick one)
    InternetService_DSL: int             = Field(0, ge=0, le=1)
    InternetService_Fiber_optic: int     = Field(0, ge=0, le=1)
    InternetService_No: int              = Field(1, ge=0, le=1)
    # Contract (one-hot — pick one)
    Contract_Month_to_month: int         = Field(1, ge=0, le=1)
    Contract_One_year: int               = Field(0, ge=0, le=1)
    Contract_Two_year: int               = Field(0, ge=0, le=1)
    # Payment (one-hot — pick one)
    PaymentMethod_Bank_transfer: int     = Field(0, ge=0, le=1)
    PaymentMethod_Credit_card: int       = Field(0, ge=0, le=1)
    PaymentMethod_Electronic_check: int  = Field(1, ge=0, le=1)
    PaymentMethod_Mailed_check: int      = Field(0, ge=0, le=1)


def build_feature_df(c: CustomerInput) -> pd.DataFrame:
    """
    Map request fields → exact feature_names from metadata.json.
    Column names with spaces/dashes must be matched carefully.
    """
    row = {
        "gender"                             : c.gender,
        "SeniorCitizen"                      : c.SeniorCitizen,
        "Partner"                            : c.Partner,
        "Dependents"                         : c.Dependents,
        "tenure"                             : c.tenure,
        "PhoneService"                       : c.PhoneService,
        "MultipleLines"                      : c.MultipleLines,
        "OnlineSecurity"                     : c.OnlineSecurity,
        "OnlineBackup"                       : c.OnlineBackup,
        "DeviceProtection"                   : c.DeviceProtection,
        "TechSupport"                        : c.TechSupport,
        "StreamingTV"                        : c.StreamingTV,
        "StreamingMovies"                    : c.StreamingMovies,
        "PaperlessBilling"                   : c.PaperlessBilling,
        "MonthlyCharges"                     : c.MonthlyCharges,
        "TotalCharges"                       : c.TotalCharges,
        # ── These names must match metadata exactly (spaces/dashes) ──
        "InternetService_DSL"                : c.InternetService_DSL,
        "InternetService_Fiber optic"        : c.InternetService_Fiber_optic,
        "InternetService_No"                 : c.InternetService_No,
        "Contract_Month-to-month"            : c.Contract_Month_to_month,
        "Contract_One year"                  : c.Contract_One_year,
        "Contract_Two year"                  : c.Contract_Two_year,
        "PaymentMethod_Bank transfer (automatic)": c.PaymentMethod_Bank_transfer,
        "PaymentMethod_Credit card (automatic)"  : c.PaymentMethod_Credit_card,
        "PaymentMethod_Electronic check"     : c.PaymentMethod_Electronic_check,
        "PaymentMethod_Mailed check"         : c.PaymentMethod_Mailed_check,
        # ── Engineered features ──
        "ChargesPerMonth"                    : c.TotalCharges / (c.tenure + 1),
        "IsLongTenure"                       : int(c.tenure > 24),
        "HighMonthlyCharge"                  : int(c.MonthlyCharges > 64.76),
    }

    df = pd.DataFrame([row])[FEATURE_NAMES]   # enforce exact column order

    # Scale numerical columns
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])
    return df


def get_shap_top5(df: pd.DataFrame):
    sv = explainer.shap_values(df)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv[0] if sv.ndim == 2 else sv
    pairs = sorted(zip(FEATURE_NAMES, sv),
                   key=lambda x: abs(x[1]), reverse=True)[:5]
    return [
        {
            "feature"   : f,
            "shap_value": round(float(v), 4),
            "direction" : "increases churn" if v > 0 else "decreases churn",
        }
        for f, v in pairs
    ]


def risk_level(p: float) -> str:
    if p < 0.30: return "low"
    if p < 0.55: return "medium"
    return "high"


def recommended_actions(p: float) -> list:
    if p < 0.30:
        return [
            "No immediate retention action needed",
            "Include in standard NPS survey",
        ]
    elif p < 0.55:
        return [
            "Schedule proactive check-in call within 7 days",
            "Offer loyalty discount or free month",
            "Propose contract upgrade consultation",
        ]
    else:
        return [
            "Priority retention call — same day",
            "Offer 1-year contract with 20% discount",
            "Add 3 months of free service add-on",
            "Flag to Customer Success Manager immediately",
        ]


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status" : "ok",
        "message": "Telco Churn Prediction API",
        "version": "1.0.0",
        "docs"   : "/docs",
    }


@app.get("/health")
def health():
    return {
        "status"          : "healthy",
        "models_loaded"   : ["lightgbm", "dnn"],
        "model_auc"       : meta["model_auc"],
        "ensemble_weights": {"lgb": LGB_W, "dnn": DNN_W},
        "threshold"       : round(THRESHOLD, 3),
        "n_features"      : len(FEATURE_NAMES),
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        df = build_feature_df(customer)

        lgb_prob = float(lgb_model.predict(df)[0])
        dnn_prob = float(dnn_model.predict(df, verbose=0)[0][0])
        ens_prob = lgb_prob * LGB_W + dnn_prob * DNN_W

        agreement = abs(lgb_prob - dnn_prob) < 0.10
        shap_top5 = get_shap_top5(df)

        return {
            "predictions": {
                "lgb_probability"      : round(lgb_prob, 4),
                "dnn_probability"      : round(dnn_prob, 4),
                "ensemble_probability" : round(ens_prob, 4),
                "lgb_weight"           : LGB_W,
                "dnn_weight"           : DNN_W,
            },
            "decision": {
                "churn_predicted"  : bool(ens_prob >= THRESHOLD),
                "threshold_used"   : round(THRESHOLD, 3),
                "risk_level"       : risk_level(ens_prob),
                "model_agreement"  : agreement,
                "disagreement_gap" : round(abs(lgb_prob - dnn_prob), 4),
            },
            "explainability": {
                "top_drivers": shap_top5,
                "model"      : "lightgbm",
            },
            "recommended_actions": recommended_actions(ens_prob),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerInput]):
    if len(customers) > 500:
        raise HTTPException(status_code=400, detail="Max 500 customers per batch")
    results = []
    for i, c in enumerate(customers):
        try:
            r = predict(c)
            r["index"] = i
            results.append(r)
        except Exception as e:
            results.append({"index": i, "error": str(e)})
    return {"count": len(results), "results": results}


@app.get("/model/info")
def model_info():
    return {
        "models": {
            "lightgbm": {"type": "gradient_boosting", "weight": LGB_W,
                         "auc": meta["model_auc"]["lgb"]},
            "dnn"     : {"type": "neural_network",    "weight": DNN_W,
                         "auc": meta["model_auc"]["dnn"]},
        },
        "ensemble_auc"    : meta["model_auc"]["ensemble"],
        "threshold"       : round(THRESHOLD, 3),
        "n_features"      : len(FEATURE_NAMES),
        "feature_names"   : FEATURE_NAMES,
        "engineered_feats": ["ChargesPerMonth", "IsLongTenure", "HighMonthlyCharge"],
    }