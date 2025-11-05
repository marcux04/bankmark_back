from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from subprocess import run, CalledProcessError

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["bank_marketing_db"]
results_collection = db["results"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "label_encoders.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ Modelo no encontrado en {MODEL_PATH}. Ejecuta train_model.py primero.")

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else {}

app = FastAPI(title="Bank Marketing API", version="1.0")


class ClientData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str


def encode_input(df_row: pd.DataFrame) -> pd.DataFrame:
    df = df_row.copy()
    for col, encoder in label_encoders.items():
        if col in df.columns:
            val = df.at[0, col]
            try:
                df[col] = encoder.transform([val])[0]
            except Exception:
                df[col] = -1
    # asegurar columnas
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        for c in model_features:
            if c not in df.columns:
                df[c] = 0
        df = df[model_features]
    return df


@app.get("/")
def root():
    return {"message": "✅ API de Random Forest - Bank Marketing"}


@app.get("/metrics")
def get_metrics():
    results = list(results_collection.find({}, {"_id": 0}).sort("timestamp", -1))
    if not results:
        raise HTTPException(status_code=404, detail="No hay métricas registradas.")
    return {"metrics": results}


@app.post("/predict")
def predict(client: ClientData):
    df = pd.DataFrame([client.dict()])
    if label_encoders:
        df_encoded = encode_input(df)
    else:
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            df[col] = pd.factorize(df[col])[0]
        df_encoded = df

    try:
        pred = int(model.predict(df_encoded)[0])
        proba = float(model.predict_proba(df_encoded)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")

    message = "El cliente contratará el servicio." if pred == 1 else "El cliente no contratará el servicio."
    return {"prediction": pred, "probability_yes": round(proba, 4), "message": message}

@app.post("/retrain")
def retrain_model():
    import subprocess
    import sys

    # Ruta absoluta al train_model.py dentro de /model/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(project_root, "model", "train_model.py")

    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail=f"No se encontró el archivo: {script_path}")

    try:
        result = subprocess.run(
            [sys.executable, script_path],  # usa el mismo intérprete que FastAPI
            capture_output=True,
            text=True,
            check=True,
            cwd=project_root  # ejecuta desde la raíz del proyecto
        )
        return {
            "message": "✅ Reentrenamiento ejecutado correctamente.",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error ejecutando el reentrenamiento: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error general: {e}")


@app.get("/ping")
def ping():
    """Verifica si la API está activa."""
    return {"message": "pong"}