import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import os

# --- Cargar variables de entorno ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# --- Conectar a MongoDB ---
try:
    client = MongoClient(MONGO_URI)
    db = client["bank_marketing_db"]
    results_collection = db["results"]
    print("[OK] Conexión a MongoDB establecida.")
except Exception as e:
    print(f"[ERROR] Error al conectar a MongoDB: {e}")
    exit(1)

# --- 1. Cargar dataset ---
DATA_PATH = "data/bank.csv"
print("[INFO] Cargando dataset...")
try:
    df = pd.read_csv(DATA_PATH, sep=";")
except FileNotFoundError:
    print(f"[ERROR] No se encontró el archivo {DATA_PATH}.")
    exit(1)
except Exception as e:
    print(f"[ERROR] Error al cargar el dataset: {e}")
    exit(1)

# --- 2. Preprocesamiento ---
print("[INFO] Preprocesando datos...")
df["y"] = df["y"].map({"yes": 1, "no": 0})

cat_cols = df.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- 3. Separar variables ---
X = df.drop("y", axis=1)
y = df["y"]

# --- 4. Dividir dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- 5. Entrenar modelo ---
print("[INFO] Entrenando modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 6. Predicciones ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --- 7. Calcular métricas ---
metrics = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

# --- 8. Guardar modelo y encoders ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
print("[OK] Modelo y codificadores guardados correctamente.")

# --- 9. Guardar métricas en MongoDB ---
try:
    results_collection.delete_many({})  # limpiar anteriores
    results_collection.insert_one(metrics)
    print("[OK] Métricas guardadas en MongoDB con éxito.")
except Exception as e:
    print(f"[ERROR] Error al guardar métricas en MongoDB: {e}")

# --- 10. Mostrar resultados ---
print("\n[RESULTADOS DEL MODELO]")
for k, v in metrics.items():
    if isinstance(v, (float, int)):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")
