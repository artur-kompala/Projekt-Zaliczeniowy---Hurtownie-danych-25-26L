import pandas as pd
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine

# Inicjalizacja aplikacji
app = FastAPI(
    title="Airbnb Pricing API",
    description="API serwujące model LightGBM do wyceny nieruchomości w NYC",
    version="1.0"
)

# Konfiguracja MLflowWW
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
REGISTRY_NAME = "InsideAirbnbPricePredictionModel"

POSTGRES_USER = os.getenv("POSTGRES_USER", os.getenv("DB_USER", "analytics_user"))
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", "secure_password_123"))
POSTGRES_HOST = os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "127.0.0.1"))
POSTGRES_PORT = os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", os.getenv("DB_NAME", "airbnb_dwh"))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

NUMERIC_COLS = [
	"host_listings_count",
	"accommodates",
	"bathrooms",
	"bedrooms",
	"minimum_nights",
	"maximum_nights",
	"availability_365",
	"number_of_reviews",
	"number_of_reviews_ltm",
	"review_scores_rating",
	"reviews_per_month",
]

CATEGORICAL_COLS = [
	"room_type",
	"property_type",
	"neighbourhood_group_cleansed",
	"neighbourhood_cleansed",
]

# Zmienna globalna przechowująca model w pamięci RAM
model = None

# Struktura danych wejściowych z domyślnymi wartościami (walidacja Pydantic)
class PropertyData(BaseModel):
    minimum_nights: int = 2
    accommodates: int = 4
    bathrooms: float = 1.0
    bedrooms: int = 1
    beds: int = 2
    latitude: float = 40.7128
    longitude: float = -74.0060
    room_type: str = "Private room"  # Opcje: "Entire home/apt", "Private room", "Shared room", "Hotel room"

def get_db_engine():
    return create_engine(DATABASE_URL)

def _try_load_latest_model(registry_name: str):
    """Próbuje załadować najnowszą wersję moWWdelu z rejestru modeli."""
    global model

    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{registry_name}'")

    if not model_versions:
        print(f"⚠️ W rejestrze modeli nie znaleziono żadnych wersji dla: {registry_name}")
        return False

    latest_versions = sorted(model_versions, key=lambda version: int(version.version), reverse=True)

    for version in latest_versions:
        model_uri = f"models:/{registry_name}/{version.version}"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"✅ Ładowanie modelu z rejestru: {registry_name}, wersja: {version.version}")
            return True
        except Exception as err:
            print(
                f"⚠️ Pomijam wersję {version.version} w rejestrze {registry_name}: "
                f"brak poprawnego artefaktu modelu ({err})"
            )

    return False


def _get_latest_registered_model_version(registry_name: str):
    """Zwraca najnowszą wersję modelu z rejestru modeli albo None."""
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{registry_name}'")

    if not model_versions:
        return None

    return max(model_versions, key=lambda version: int(version.version))

@app.on_event("startup")
def load_latest_model():
    """Uruchamia się przy starcie serwera, pobierając najnowszy model z MLflow."""
    global model
    print(f"Szukanie najnowszego modelu w rejestrze MLflow: {REGISTRY_NAME}...")

    if not _try_load_latest_model(REGISTRY_NAME):
        raise RuntimeError(f"Nie udało się załadować modelu z rejestru: {REGISTRY_NAME}.")


@app.post("/predict")
def predict_price(data: PropertyData):
    """Główny endpoint do predykcji cen."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model ML nie jest gotowy.")

    # Przekształcenie danych z JSON na format oczekiwany przez model (9 cech)

    #df_encoded_features = pd.get_dummies(df_features, columns=CATEGORICAL_COLS, dtype=int)
    # Pominęliśmy 'Hotel room', którego nie było w danych treningowych w NYC
    input_data = {
        "room_type" : ["Private room"],
		"property_type" : ["Entire rental unit"],
		"neighbourhood_group_cleansed" : ["Manhattan"],
		"neighbourhood_cleansed" : ["Williamsburg"],
		"host_listings_count" : [1],
		"accommodates" : [data.accommodates],
		"bathrooms" : [data.bathrooms],
		"bedrooms" : [data.bedrooms],
		"minimum_nights" : [data.minimum_nights],
		"maximum_nights" : [365],
		"availability_365" : [365],
		"number_of_reviews" : [0],
		"number_of_reviews_ltm" : [0],
		"review_scores_rating" : [0.0],
		"reviews_per_month" : [0.0]

    }

    df_features = pd.DataFrame(input_data)

    try:
        prediction = model.predict(df_features)
        return {
            "suggested_price": round(float(prediction[0]), 2),
            "currency": "USD"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Błąd dopasowania cech: {str(e)}")

# Pobranie danych z serwera
# NOTE: Może być ciężkie obliczeniowo, ze względu na ilość danych.
@app.get("/data/listings")
def get_listings_data(offset: int = Query(default=0, ge=0), limit: int = Query(default=25, ge=1, le=100)):
    """Endpoint do pobrania pierwszych N rekordów danych treningowych, uwzględniając offset."""
    with get_db_engine().begin() as connection:
        df = pd.read_sql("SELECT * FROM clean_data LIMIT %s OFFSET %s", con=connection, params=(limit, offset))
        records = df.to_dict(orient="records")

    return {
        "count": len(records),
        "offset": offset,
        "limit": limit,
        "records": records,
    }

@app.get("/data/listings/geo")
def get_geo_data():
    """Endpoint do pobrania danych geograficznych z MLflow."""
    latest_version = _get_latest_registered_model_version(REGISTRY_NAME)

    if latest_version is None:
        raise HTTPException(status_code=404, detail=f"Nie znaleziono modelu w rejestrze: {REGISTRY_NAME}.")

    run_id = latest_version.run_id
    artifact_uri = f"runs:/{run_id}/data/listings_full.csv"

    return {
        "message": "Dane geograficzne można pobrać z MLflow.",
        "artifact_uri": artifact_uri
    }

@app.get("/model/info")
def get_current_model_info():
    """Endpoint do pobrania informacji o aktualnie załadowanym modelu."""
    latest_version = _get_latest_registered_model_version(REGISTRY_NAME)

    if latest_version is None:
        raise HTTPException(status_code=404, detail=f"Nie znaleziono modelu w rejestrze: {REGISTRY_NAME}.")

    return {
        "model_id": latest_version.run_id,
        "version": latest_version.version,
        "status": latest_version.status,
        "start_time": str(latest_version.creation_timestamp),
        "end_time": str(latest_version.last_updated_timestamp),
    }

@app.get("/webhook/new-model-available")
def new_model_available_webhook():
    """Webhook do ręcznego odświeżania modelu w pamięci RAM (np. po deployu nowej wersji)."""
    if _try_load_latest_model(REGISTRY_NAME):
        return {"message": "Model został odświeżony z rejestru."}
    else:
        raise HTTPException(status_code=500, detail="Nie udało się odświeżyć modelu z rejestru.")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2137)

if __name__ == "__main__":
    main()