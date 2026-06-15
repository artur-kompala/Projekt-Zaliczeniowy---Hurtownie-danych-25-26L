import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn

# Inicjalizacja aplikacji
app = FastAPI(
    title="Airbnb Pricing API",
    description="API serwujące model LightGBM do wyceny nieruchomości w NYC",
    version="1.0"
)

# Konfiguracja MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "Airbnb_NYC_Price_Prediction"
DATA_PATH = os.getenv("TRAINING_DATA_PATH", "../data/listings_full.csv")  # Ścieżka do danych treningowych

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


def train_model_and_save(test_size=0.2, random_state=42):
    """Funkcja pomocnicza do wytrenowania pierwszego modelu, jeśli nie ma żadnych modeli w MLflow."""
    raise NotImplementedError()


def _try_load_latest_model(experiment_id: str):
    """Próbuje załadować najnowszy poprawny model z kolejnych runów."""
    global model

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=25,
    )

    for _, run in runs.iterrows():
        run_id = run.run_id
        model_uri = f"runs:/{run_id}/model"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"✅ Ładowanie modelu z RUN_ID: {run_id}")
            return True
        except Exception as err:
            print(f"⚠️ Pomijam RUN_ID {run_id}: brak poprawnego artefaktu modelu ({err})")

    return False

@app.on_event("startup")
def load_latest_model():
    """Uruchamia się przy starcie serwera, pobierając najnowszy model z MLflow."""
    global model
    print("Szukanie najnowszego modelu w eksperymencie MLflow...")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print("Błąd: Nie znaleziono eksperymentu w MLflow.")
        mlflow.create_experiment(EXPERIMENT_NAME)
        train_model_and_save()
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if not _try_load_latest_model(experiment.experiment_id):
        print("⚠️ Błąd: Nie znaleziono wytrenowanych modeli. Rozpoczynam trening pierwszego modelu...")
        train_model_and_save()
        if not _try_load_latest_model(experiment.experiment_id):
            raise RuntimeError("Nie udało się załadować modelu po treningu.")


@app.get("/predict")
def predict_price(data: PropertyData):
    """Główny endpoint do predykcji cen."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model ML nie jest gotowy.")

    # Przekształcenie danych z JSON na format oczekiwany przez model (9 cech)
    # Pominęliśmy 'Hotel room', którego nie było w danych treningowych w NYC
    input_data = {
        'minimum_nights': [data.minimum_nights],
        'accommodates': [data.accommodates],
        'bathrooms': [data.bathrooms],
        'bedrooms': [data.bedrooms],
        'beds': [data.beds],
        'latitude': [data.latitude],
        'longitude': [data.longitude],
        'room_type_Private room': [1 if data.room_type == "Private room" else 0],
        'room_type_Shared room': [1 if data.room_type == "Shared room" else 0]
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
def get_listings_data():
    """Endpoint do pobrania danych z MLflow, aby umożliwić użytkownikowi pobranie danych treningowych."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise HTTPException(status_code=404, detail="Nie znaleziono eksperymentu w MLflow.")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)

    if runs.empty:
        raise HTTPException(status_code=404, detail="Nie znaleziono wytrenowanych modeli.")

    run_id = runs.iloc[0].run_id
    artifact_uri = f"runs:/{run_id}/data/listings_full.csv"

    return {
        "message": "Dane treningowe można pobrać z MLflow.",
        "artifact_uri": artifact_uri
    }

@app.get("/data/listings/geo")
def get_geo_data():
    """Endpoint do pobrania danych geograficznych z MLflow."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise HTTPException(status_code=404, detail="Nie znaleziono eksperymentu w MLflow.")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)

    if runs.empty:
        raise HTTPException(status_code=404, detail="Nie znaleziono wytrenowanych modeli.")

    run_id = runs.iloc[0].run_id
    artifact_uri = f"runs:/{run_id}/data/listings_full.csv"

    return {
        "message": "Dane geograficzne można pobrać z MLflow.",
        "artifact_uri": artifact_uri
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2137)

if __name__ == "__main__":
    main()