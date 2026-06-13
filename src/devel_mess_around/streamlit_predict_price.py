"""To jest przykładowa apka, która wykorzystuje dashboard streamlit do predykcji ceny
na podstawie cech listingów. Nie jest to część głównego projektu, ale może posłużyć
jako inspiracja lub punkt wyjścia do dalszych eksperymentów. Wybrałem parę cech, ale
myślę, że można się także pozbyć tych związanych z ocenami ofert.

Dane w repo nie miały wielu kolumn, więc pobrałem pełną wersję z Inside Airbnb,
żeby mieć lepszy wgląd jaki model można zbudować. Nie pushowałem zbioru danych,
ze względu na jego rozmiar. Można go pobrać z
https://data.insideairbnb.com/united-states/ny/new-york-city/2026-04-14/data/listings.csv.gz

Plik jest spakowany, więc trzeba go rozpakować przed użyciem. Po rozpakowaniu,
można go umieścić w katalogu data/ i zaktualizować ścieżkę w DATA_PATH."""

from pathlib import Path
import re

import pandas as pd
import streamlit as st
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "listings_full.csv"
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


@st.cache_data
def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Extract bathroom count as the first number found in bathrooms_text.
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


@st.cache_resource
def train_model(df_clean: pd.DataFrame):
    encoded_df = pd.get_dummies(df_clean, columns=CATEGORICAL_COLS, dtype=int)

    X = encoded_df.drop(columns=[TARGET_COL])
    y = encoded_df[TARGET_COL]

    raw_feature_columns = X.columns.tolist()
    feature_name_map = build_safe_feature_name_map(raw_feature_columns)
    X = X.rename(columns=feature_name_map)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    return model, raw_feature_columns, feature_name_map, metrics


def build_safe_feature_name_map(columns: list[str]) -> dict[str, str]:
    safe_names: list[str] = []
    used_names: set[str] = set()

    for col in columns:
        # LightGBM rejects feature names containing JSON-reserved characters.
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


def build_input_row(
    df_clean: pd.DataFrame,
    model_columns: list[str],
    feature_name_map: dict[str, str],
) -> pd.DataFrame:
    input_values: dict[str, object] = {}

    st.subheader("Listing Features")

    col1, col2 = st.columns(2)

    with col1:
        for col in NUMERIC_COLS[: len(NUMERIC_COLS) // 2]:
            default_val = float(df_clean[col].median())
            min_val = float(df_clean[col].min())
            max_val = float(df_clean[col].max())
            input_values[col] = st.number_input(
                label=col.replace("_", " ").title(),
                min_value=min_val,
                max_value=max_val,
                value=default_val,
            )

    with col2:
        for col in NUMERIC_COLS[len(NUMERIC_COLS) // 2 :]:
            default_val = float(df_clean[col].median())
            min_val = float(df_clean[col].min())
            max_val = float(df_clean[col].max())
            input_values[col] = st.number_input(
                label=col.replace("_", " ").title(),
                min_value=min_val,
                max_value=max_val,
                value=default_val,
            )

    st.subheader("Categorical Features")
    cat_col1, cat_col2 = st.columns(2)
    category_columns = [cat_col1, cat_col2]

    for idx, col in enumerate(CATEGORICAL_COLS):
        options = sorted(df_clean[col].dropna().unique().tolist())
        with category_columns[idx % 2]:
            input_values[col] = st.selectbox(
                label=col.replace("_", " ").title(),
                options=options,
                index=0,
            )

    input_df = pd.DataFrame([input_values])
    input_df = pd.get_dummies(input_df, columns=CATEGORICAL_COLS, dtype=int)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    input_df = input_df.rename(columns=feature_name_map)

    return input_df


def main():
    st.set_page_config(page_title="Price Prediction", layout="wide")
    st.title("Nightly Price Prediction")
    st.caption("Train a regression model and estimate Airbnb price from listing features.")

    df_clean = load_and_prepare_data()
    model, model_columns, feature_name_map, metrics = train_model(df_clean)

    m1, m2, m3 = st.columns(3)
    m1.metric("Rows used", f"{len(df_clean):,}")
    m2.metric("Validation MAE", f"{metrics['mae']:.2f}")
    m3.metric("Validation R2", f"{metrics['r2']:.3f}")

    input_df = build_input_row(df_clean, model_columns, feature_name_map)

    if st.button("Predict Price", type="primary"):
        prediction = float(model.predict(input_df)[0])
        st.success(f"Predicted nightly price: {prediction:.2f}")


if __name__ == "__main__":
    main()