import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow

# Inicjalizacja aplikacji
app = FastAPI(
    title="Airbnb Pricing API",
    description="API serwujące model LightGBM do wyceny nieruchomości w NYC",
    version="1.0"
)

# Konfiguracja MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "Airbnb_NYC_Price_Prediction"

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


@app.on_event("startup")
def load_latest_model():
    """Uruchamia się przy starcie serwera, pobierając najnowszy model z MLflow."""
    global model
    print("Szukanie najnowszego modelu w eksperymencie MLflow...")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print("Błąd: Nie znaleziono eksperymentu w MLflow.")
        return

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)

    if not runs.empty:
        run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{run_id}/model"
        print(f"✅ Ładowanie modelu z RUN_ID: {run_id}")
        model = mlflow.pyfunc.load_model(model_uri)
    else:
        print("⚠️ Błąd: Nie znaleziono wytrenowanych modeli.")


@app.post("/predict")
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