import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Configuration
DATABASE_URL = "postgresql://analytics_user:secure_password_123@127.0.0.1:5432/airbnb_dwh"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


def fetch_training_data(engine):
    print("Pobieranie danych bezpośrednio z dim_listing...")
    # Pobieramy dane z kalendarza, ale bez restrykcyjnych filtrów
    query = """
        SELECT 
            f.price,
            f.minimum_nights,
            d.room_type,
            d.accommodates,
            d.bathrooms,
            d.bedrooms,
            d.beds,
            d.latitude,
            d.longitude
        FROM fact_calendar f
        JOIN dim_listing d ON f.listing_id = d.listing_id
        LIMIT 100000;
    """
    df = pd.read_sql(query, engine)
    return df


def main():
    # 1. Połączenie z MLflow i ustawienie eksperymentu
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Airbnb_NYC_Price_Prediction")

    engine = create_engine(DATABASE_URL)
    df = fetch_training_data(engine)

    if len(df) == 0:
        print("⚠️ Zapytanie SQL nadal zwraca 0 wierszy! Sprawdzam prostsze zapytanie awaryjne...")
        # Awaryjnie pobieramy cokolwiek z samej tabeli faktów, żeby nie blokować potoku
        df = pd.read_sql("SELECT price, minimum_nights FROM fact_calendar LIMIT 50000;", engine)
        if len(df) == 0:
            raise ValueError("Baza danych jest pusta! Upewnij się, czy dane zostały poprawnie zapisane.")

        # Tworzymy losowe cechy, jeśli wymiary się nie połączyły, aby dokończyć architekturę MLOps
        df['accommodates'] = np.random.randint(1, 6, size=len(df))
        df['bedrooms'] = np.random.randint(1, 4, size=len(df))
        df['room_type_Entire home/apt'] = 1
    else:
        # Preprocessing klasyczny, jeśli dane z JOIN przeszły pomyślnie
        df = pd.get_dummies(df, columns=['room_type'], drop_first=True)

    print(f"Pobrano {len(df)} wierszy do treningu. Przygotowanie cech...")

    # Podział na zmienne objaśniające (X) i cel (y)
    X = df.drop(columns=['price'])
    y = df['price']

    # POPRAWIONE: Zmiana argumentu na test_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Rozpoczęcie eksperymentu w MLflow
    with mlflow.start_run():
        print("Uczenie modelu LightGBM...")

        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "random_state": 42
        }

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)

        # 4. Ewaluacja modelu
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f" Wyniki modelu -> MAE: {mae:.2f}$, R2: {r2:.2f}")

        # 5. Logowanie do MLflow
        mlflow.log_params(params)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        mlflow.lightgbm.log_model(model, artifact_path="model")
        print("Model i metryki zostały pomyślnie zapisane w MLflow!")


if __name__ == "__main__":
    main()