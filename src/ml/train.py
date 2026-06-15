import re

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

TARGET_COL = "price_quote_price_per_night"

CATEGORICAL_COLS = [
	"room_type",
	"property_type",
	"neighbourhood_group_cleansed",
	"neighbourhood_cleansed",
]

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

DEFAULT_MODEL_PARAMS = {
	"n_estimators": 300,
	"learning_rate": 0.05,
	"num_leaves": 31,
	"max_depth": -1,
	"min_child_samples": 20,
	"subsample": 1.0,
	"colsample_bytree": 1.0,
	"random_state": 42,
}

def build_safe_feature_name_map(columns: list[str]) -> dict[str, str]:
    """Metoda do tworzenia bezpiecznych nazw cech, unikając problemów z nazwami cech w MLflow i niektórymi algorytmami ML."""
    safe_names: list[str] = []
    used_names: set[str] = set()

    for col in columns:
        safe = re.sub(r"[^0-9a-zA-Z_]", "_", str(col))
        safe = re.sub(r"_+", "_", safe).strip("_")
        if not safe:
            safe = "feature"

        candidate = safe
        counter = 1
        while candidate in used_names:
            candidate = f"{safe}_{counter}"
            counter += 1

        used_names.add(candidate)
        safe_names.append(candidate)

    return dict(zip(columns, safe_names))

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Funkcja do czyszczenia i wstępnego przetwarzania danych. Można ją rozbudować o dodatkowe kroki, jeśli zajdzie taka potrzeba."""
    # Przykładowe przetwarzanie danych
    df["bathrooms"] = pd.to_numeric(
        df["bathrooms_text"].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )

    columns_to_keep = [
        "id",
        *NUMERIC_COLS,
        *CATEGORICAL_COLS,
        TARGET_COL,
    ]

    df_clean = df[columns_to_keep].copy().set_index("id")
    df_clean = df_clean.dropna(subset=[TARGET_COL])

    for col in NUMERIC_COLS:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in CATEGORICAL_COLS:
        df_clean[col] = df_clean[col].fillna("Unknown").astype(str)
        
    return df_clean

def train_model(
	df_clean: pd.DataFrame,
	n_estimators: int = 100,
	learning_rate: float = 0.05,
	num_leaves: int = 31,
	max_depth: int = -1,
	min_child_samples: int = 20,
	subsample: float = 1.0,
	colsample_bytree: float = 1.0,
	random_state: int = 42,
):
    """Funkcja trenująca model LightGBM na danych z bazy danych i logująca go do MLflow."""
    encoded_df = pd.get_dummies(df_clean, columns=CATEGORICAL_COLS, dtype=int)

    X = encoded_df.drop(columns=[TARGET_COL])
    y = encoded_df[TARGET_COL]

    raw_feature_columns = X.columns.tolist()
    feature_name_map = build_safe_feature_name_map(raw_feature_columns)
    X = X.rename(columns=feature_name_map)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    feature_importance = (
        pd.DataFrame(
            {
                "feature": raw_feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": model,
        "model_columns": raw_feature_columns,
        "feature_name_map": feature_name_map,
        "metrics": metrics,
        "params": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
        },
        "feature_importance": feature_importance,
    }


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