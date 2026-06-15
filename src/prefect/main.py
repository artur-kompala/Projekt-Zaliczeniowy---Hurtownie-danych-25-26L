import os

import requests
from sklearn.model_selection import train_test_split

from prefect import task, flow
import pandas as pd
import re
import io
import gzip
import mlflow
from mlflow import log_metric, log_param, log_artifact
import re
import pandas as pd
from sqlalchemy import create_engine
import mlflow
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
INSIDE_AIRBNB_DATA_LANDING_PAGE = os.getenv("INSIDE_AIRBNB_DATA_LANDING_PAGE", "https://insideairbnb.com/get-the-data.html")
POSTGRES_USER = os.getenv("POSTGRES_USER", os.getenv("DB_USER", "analytics_user"))
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", "secure_password_123"))
POSTGRES_HOST = os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "127.0.0.1"))
POSTGRES_PORT = os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", os.getenv("DB_NAME", "airbnb_dwh"))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

RAW_DATA_TABLE = os.getenv("RAW_DATA_TABLE", "raw_data")
CLEAN_DATA_TABLE = os.getenv("CLEAN_DATA_TABLE", "clean_data")

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

PATTERN = re.compile(
    r"(https://data\.insideairbnb\.com/united-states/ny/new-york-city/"
    r"(?P<ymd_date>\d{4}-\d{2}-\d{2})/data/listings\.csv\.gz)"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_db_engine():
    return create_engine(DATABASE_URL)

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


@task
def fetch_data_from_source(link_to_gzip_csv: str) -> pd.DataFrame:
    """Zadanie do pobierania danych z linku Inside Airbnb."""
    if not link_to_gzip_csv or not isinstance(link_to_gzip_csv, str):
        raise ValueError("Nieprawidłowy link do archiwum gzip CSV.")

    try:
        response = requests.get(link_to_gzip_csv, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Nie udało się pobrać danych z linku: {link_to_gzip_csv}") from exc

    content_type = (response.headers.get("Content-Type") or "").lower()
    if "gzip" not in content_type and not link_to_gzip_csv.endswith(".gz"):
        raise ValueError(
            f"Nieprawidłowy format danych: '{response.headers.get('Content-Type')}'. Oczekiwano archiwum gzip."
        )

    try:
        decompressed_bytes = gzip.decompress(response.content)
    except (OSError, EOFError) as exc:
        raise RuntimeError("Nie udało się rozpakować archiwum gzip.") from exc

    try:
        csv_buffer = io.StringIO(decompressed_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise RuntimeError("Nie udało się zdekodować danych CSV do UTF-8.") from exc

    try:
        df = pd.read_csv(csv_buffer)
    except (pd.errors.ParserError, ValueError) as exc:
        raise RuntimeError("Nie udało się sparsować pliku CSV.") from exc

    if df.empty:
        raise ValueError("Pobrany plik CSV jest pusty.")

    return df

@task
def scrape_listings_url_and_date(main_url: str) -> tuple[str, str]:
    """Funkcja do znalezienia linku do danych i daty z tekstu strony Inside Airbnb."""
    
    response = requests.get(main_url, timeout=10)
    response.raise_for_status()
    text = response.text

    match = PATTERN.search(text)
    if not match:
        raise ValueError(
            "Missing URL: https://data.insideairbnb.com/united-states/ny/new-york-city/{YMD_date}/data/listings.csv.gz"
        )

    found_url = match.group(1)
    ymd_date = match.group("ymd_date")
    return found_url, ymd_date

@task
def save_raw_data_to_db(df: pd.DataFrame):
    """Zadanie do zapisywania surowych danych do bazy danych."""
    if df.empty:
        raise ValueError("Nie można zapisać pustego zbioru surowych danych do bazy.")

    with get_db_engine().begin() as connection:
        df.to_sql(RAW_DATA_TABLE, connection, if_exists="replace", index=False, method="multi")

@task
def clean_and_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Zadanie czyszczenia i wstępnego przetwarzania danych."""
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

@task
def save_clean_data_to_db(df_clean: pd.DataFrame):
    """Zadanie do zapisywania przetworzonych danych do bazy danych."""
    if df_clean.empty:
        raise ValueError("Nie można zapisać pustego zbioru przetworzonych danych do bazy.")

    df_to_save = df_clean.reset_index()

    with get_db_engine().begin() as connection:
        df_to_save.to_sql(CLEAN_DATA_TABLE, connection, if_exists="replace", index=False, method="multi")

@task
def get_clean_data_from_db() -> pd.DataFrame:
    """Zadanie do pobierania przetworzonych danych z bazy danych."""
    with get_db_engine().connect() as connection:
        df_clean = pd.read_sql_table(CLEAN_DATA_TABLE, connection)

    if "id" in df_clean.columns:
        df_clean = df_clean.set_index("id")

    return df_clean

@task
def train_model(X_train, X_test, y_train, y_test, n_estimators=100, raw_feature_columns=[], feature_name_map=[]) -> LGBMRegressor:
    """Funkcja trenująca model LightGBM na danych z bazy danych i logująca go do MLflow."""

    model = LGBMRegressor(
        n_estimators=n_estimators,
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
            "learning_rate": model.learning_rate,
            "num_leaves": model.num_leaves,
            "max_depth": model.max_depth,
            "min_child_samples": model.min_child_samples,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
            "random_state": model.random_state,
        },
        "feature_importance": feature_importance,
    }

@task
def split_data(df: pd.DataFrame):
    """Funkcja do podziału danych na zbiór treningowy i testowy. Używana w Prefect do zautomatyzowania procesu."""
    encoded_df = pd.get_dummies(df, columns=CATEGORICAL_COLS, dtype=int)

    X = encoded_df.drop(columns=[TARGET_COL])
    y = encoded_df[TARGET_COL]

    raw_feature_columns = X.columns.tolist()
    feature_name_map = build_safe_feature_name_map(raw_feature_columns)
    X = X.rename(columns=feature_name_map)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
    )

    return X_train, X_test, y_train, y_test, raw_feature_columns, feature_name_map

@task
def save_model_to_mlflow(training_result):
    """Zadanie do zapisywania modelu do MLflow."""
    model = training_result["model"]

    with mlflow.start_run() as run:
        mlflow.log_params(training_result["params"])
        mlflow.log_metrics(training_result["metrics"])
        mlflow.sklearn.log_model(model, "model")
        mlflow.register_model(f"runs:/{run.info.run_id}/model", "InsideAirbnbPricePredictionModel")

@flow(name="Inside AirBnB New York Price Prediction Flow")
def airbnb_new_york_price_prediction_full_pipeline():
    """Główny przepływ Prefect do automatyzacji procesu predykcji cen w Airbnb w Nowym Jorku."""
    link, date = scrape_listings_url_and_date(INSIDE_AIRBNB_DATA_LANDING_PAGE)
    df_raw = fetch_data_from_source(link)
    save_raw_data_to_db.submit(df_raw)
    df_clean = clean_and_preprocess_data(df_raw)
    save_clean_data_to_db.submit(df_clean)

    X_train, X_test, y_train, y_test, raw_feature_columns, feature_name_map = split_data(df_clean)
    model = train_model(X_train, X_test, y_train, y_test, raw_feature_columns=raw_feature_columns, feature_name_map=feature_name_map)
    save_model_to_mlflow.submit(model)

@flow(name="Inside AirBnB New York Price Prediction Retraining Flow")
def airbnb_new_york_price_prediction_retraining_flow():
    """Przepływ Prefect do ponownego trenowania modelu predykcji cen w Airbnb w Nowym Jorku.
    Bez etapów pobierania surowych danych i ich czyszczenia, zakładając, że dane są już
    dostępne w bazie danych."""

    df_clean = get_clean_data_from_db()
    X_train, X_test, y_train, y_test, raw_feature_columns, feature_name_map = split_data(df_clean)
    model = train_model(X_train, X_test, y_train, y_test, raw_feature_columns=raw_feature_columns, feature_name_map=feature_name_map)
    save_model_to_mlflow(model)


print("Uruchamianie głównego przepływu Prefect do predykcji cen Airbnb w Nowym Jorku...")
airbnb_new_york_price_prediction_full_pipeline()