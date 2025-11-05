from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Conectar a MongoDB
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

# Crear base de datos
db = client["bank_marketing_db"]

# Colección donde guardaremos métricas
results_collection = db["results"]

print("✅ Conexión a MongoDB establecida correctamente.")
