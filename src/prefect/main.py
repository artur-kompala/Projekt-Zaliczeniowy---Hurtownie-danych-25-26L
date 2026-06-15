import requests
from sklearn.model_selection import train_test_split

from prefect import task, flow
import pandas as pd
import re
import io
import gzip
import mlflow
from mlflow import log_metric, log_param, log_artifact

INSIDE_AIRBNB_DATA_LANDING_PAGE = "https://insideairbnb.com/get-the-data.html"

PATTERN = re.compile(
    r"(https://data\.insideairbnb\.com/united-states/ny/new-york-city/"
    r"(?P<ymd_date>\d{4}-\d{2}-\d{2})/data/listings\.csv\.gz)"
)

@task
def get_data_from_link(link_to_gzip_csv: str) -> pd.DataFrame:
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
def scrape_listings_url_and_date(text: str) -> tuple[str, str]:
    """Funkcja do znalezienia linku do danych i daty z tekstu strony Inside Airbnb."""
    match = PATTERN.search(text)
    if not match:
        raise ValueError(
            "Missing URL: https://data.insideairbnb.com/united-states/ny/new-york-city/{YMD_date}/data/listings.csv.gz"
        )

    found_url = match.group(1)
    ymd_date = match.group("ymd_date")
    return found_url, ymd_date

@task
def fetch_raw_data_from_source():
    """Funkcja do pobierania surowych danych z bazy danych. Używana w Prefect do zautomatyzowania procesu."""
    engine = create_engine(DATABASE_URL)
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

@task
def save_raw_data_to_db():
    """Zadanie do zapisywania surowych danych do bazy danych."""
    pass

@task
def clean_and_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Zadanie czyszczenia i wstępnego przetwarzania danych surowych."""
    pass

@task
def save_clean_data_to_db():
    """Zadanie do zapisywania przetworzonych danych do bazy danych."""
    pass

@task
def get_clean_data_from_db() -> pd.DataFrame:
    """Zadanie do pobierania przetworzonych danych z bazy danych."""
    pass

@task
def train_model(X_train, X_test, y_train, y_test, n_estimators=100):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42
    )

    with mlflow.start_run():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        mlflow.log_param(
            "n_estimators",
            n_estimators
        )

        mlflow.log_metric(
            "mse",
            mse
        )

        print("MSE:", mse)

    return model

@task
def split_data(df: pd.DataFrame):
    """Funkcja do podziału danych na zbiór treningowy i testowy. Używana w Prefect do zautomatyzowania procesu."""
    X = df.drop(columns=['price'])
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@task
def save_model_to_mlflow(model):
    """Zadanie do zapisywania modelu do MLflow."""
    mlflow.sklearn.log_model(model, "model")

@flow(name="Inside AirBnB New York Price Prediction Flow")
def airbnb_new_york_price_prediction_full_pipeline():
    """Główny przepływ Prefect do automatyzacji procesu predykcji cen w Airbnb w Nowym Jorku."""
    link, date = scrape_listings_url_and_date()
    df_raw = get_data_from_link(link)
    save_raw_data_to_db(df_raw)
    df_clean = clean_and_preprocess_data(df_raw)
    save_clean_data_to_db(df_clean)

    X_train, X_test, y_train, y_test = split_data(df_clean)
    model = train_model(X_train, X_test, y_train, y_test)
    save_model_to_mlflow(model)

@flow(name="Inside AirBnB New York Price Prediction Retraining Flow")
def airbnb_new_york_price_prediction_retraining_flow():
    """Przepływ Prefect do ponownego trenowania modelu predykcji cen w Airbnb w Nowym Jorku.
    Bez etapów pobierania surowych danych i ich czyszczenia, zakładając, że dane są już
    dostępne w bazie danych."""

    df_clean = get_clean_data_from_db()
    X_train, X_test, y_train, y_test = split_data(df_clean)
    model = train_model(X_train, X_test, y_train, y_test)