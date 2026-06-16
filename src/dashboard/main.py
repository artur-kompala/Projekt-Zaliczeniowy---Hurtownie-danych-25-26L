import os
from pathlib import Path

import pandas as pd
import streamlit as st
from api_middleware import ApiMiddleware

API_URL = os.getenv("API_URL", "http://localhost:2137")
api: ApiMiddleware = ApiMiddleware(api_url=API_URL)

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
	if not DATA_PATH.exists():
		raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

	df = pd.read_csv(DATA_PATH)

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

def build_prediction_payload(df_clean: pd.DataFrame, key_prefix: str) -> dict:
	st.subheader("Dane wejściowe predykcji")

	c1, c2 = st.columns(2)

	with c1:
		minimum_nights = st.number_input(
			"Minimum nights",
			min_value=1,
			max_value=365,
			value=int(df_clean["minimum_nights"].median()),
			step=1,
			key=f"{key_prefix}_minimum_nights",
		)
		accommodates = st.number_input(
			"Accommodates",
			min_value=1,
			max_value=16,
			value=int(df_clean["accommodates"].median()),
			step=1,
			key=f"{key_prefix}_accommodates",
		)
		bathrooms = st.number_input(
			"Bathrooms",
			min_value=0.0,
			max_value=10.0,
			value=float(df_clean["bathrooms"].median()),
			step=0.5,
			key=f"{key_prefix}_bathrooms",
		)
		bedrooms = st.number_input(
			"Bedrooms",
			min_value=0,
			max_value=20,
			value=int(df_clean["bedrooms"].median()),
			step=1,
			key=f"{key_prefix}_bedrooms",
		)

		host_listings_count = st.number_input(
			"Host listings count",
			min_value=0,
			max_value=100,
			value=int(df_clean["host_listings_count"].median()),
			step=1,
			key=f"{key_prefix}_host_listings_count",
		)
		maximum_nights = st.number_input(
			"Maximum nights",
			min_value=1,
			max_value=365,
			value=int(df_clean["maximum_nights"].median()),
			step=1,
			key=f"{key_prefix}_maximum_nights",
		)
		availability_365 = st.number_input(
			"Availability 365",
			min_value=0,
			max_value=365,
			value=int(df_clean["availability_365"].median()),
			step=1,
			key=f"{key_prefix}_availability_365",
		)
		number_of_reviews = st.number_input(
			"Number of reviews",
			min_value=0,
			max_value=1000,
			value=int(df_clean["number_of_reviews"].median()),
			step=1,
			key=f"{key_prefix}_number_of_reviews",
		)
		number_of_reviews_ltm = st.number_input(
			"Number of reviews LTM",
			min_value=0,
			max_value=1000,
			value=int(df_clean["number_of_reviews_ltm"].median()),
			step=1,
			key=f"{key_prefix}_number_of_reviews_ltm",
		)
		review_scores_rating = st.number_input(
			"Review scores rating",
			min_value=0.0,
			max_value=100.0,
			value=float(df_clean["review_scores_rating"].median()),
			step=0.5,
			key=f"{key_prefix}_review_scores_rating",
		)
		reviews_per_month = st.number_input(
			"Reviews per month",
			min_value=0.0,
			max_value=100.0,
			value=float(df_clean["reviews_per_month"].median()),
			step=0.1,
			key=f"{key_prefix}_reviews_per_month",
		)

	with c2:
		categories = api.get_categories()
		for key, value in categories.items():
			if key == "room_type":
				room_type = st.selectbox(
					"Room type",
					options=value,
					index=1,
					key=f"{key_prefix}_room_type",
				)
			elif key == "property_type":
				property_type = st.selectbox(
					"Property type",
					options=value,
					index=0,
					key=f"{key_prefix}_property_type",
				)
			elif key == "neighbourhood_group_cleansed":
				neighbourhood_group_cleansed = st.selectbox(
					"Neighbourhood group",
					options=value,
					index=0,
					key=f"{key_prefix}_neighbourhood_group_cleansed",
				)
			elif key == "neighbourhood_cleansed":
				neighbourhood_cleansed = st.selectbox(
					"Neighbourhood",
					options=value,
					index=0,
					key=f"{key_prefix}_neighbourhood_cleansed",
				)

	return {
		"room_type" : room_type,
		"property_type" : property_type,
		"neighbourhood_group_cleansed" : neighbourhood_group_cleansed,
		"neighbourhood_cleansed" : neighbourhood_cleansed,

		"host_listings_count" : int(host_listings_count),
		"accommodates" : int(accommodates),
		"bathrooms" : float(bathrooms),
		"bedrooms" : int(bedrooms),
		"minimum_nights" : int(minimum_nights),
		"maximum_nights" : int(maximum_nights),
		"availability_365" : int(availability_365),
		"number_of_reviews" : int(number_of_reviews),
		"number_of_reviews_ltm" : int(number_of_reviews_ltm),
		"review_scores_rating" : float(review_scores_rating),
		"reviews_per_month" : float(reviews_per_month)
	}


def get_default_browse_columns() -> list[str]:
	return [TARGET_COL, *NUMERIC_COLS[:4], *CATEGORICAL_COLS[:2]]


def fetch_server_listings_for_limit() -> None:
	try:
		limit = int(st.session_state["server_limit"])
		response = api.get_listings_data(limit=limit)
		records = response.get("records", [])
		st.session_state["server_listings_df"] = pd.DataFrame(records)
		st.session_state["server_listings_error"] = None
	except Exception as exc:
		st.session_state["server_listings_df"] = pd.DataFrame()
		st.session_state["server_listings_error"] = str(exc)


def render_dataset_browser(df_clean: pd.DataFrame) -> None:
	st.header("Przegląd danych")
	st.caption("Przeglądaj dane pobrane z API. Pobranie następuje po puszczeniu suwaka.")

	if "server_limit" not in st.session_state:
		st.session_state["server_limit"] = 25

	if "server_listings_df" not in st.session_state:
		fetch_server_listings_for_limit()

	st.slider(
		"Liczba rekordów do pobrania z serwera",
		min_value=1,
		max_value=100,
		value=int(st.session_state["server_limit"]),
		step=1,
		key="server_limit",
		on_change=fetch_server_listings_for_limit,
	)

	server_error = st.session_state.get("server_listings_error")
	if server_error:
		st.error(f"Błąd pobierania danych z API: {server_error}")
		active_df = df_clean
		st.info("Wyświetlane są dane lokalne jako fallback.")
	else:
		server_df = st.session_state.get("server_listings_df", pd.DataFrame())
		active_df = server_df if not server_df.empty else df_clean

	all_columns = active_df.columns.tolist()
	default_columns = [col for col in get_default_browse_columns() if col in all_columns]

	selected_columns = st.multiselect(
		"Kolumny do wyświetlenia",
		options=all_columns,
		default=default_columns,
	)
	if not selected_columns:
		st.warning("Wybierz przynajmniej jedną kolumnę do przeglądania zbioru danych.")
		return

	max_rows = max(len(active_df), 1)
	row_limit = 100

	view_df = active_df[selected_columns]

	m1, m2, m3 = st.columns(3)
	m1.metric("Wiersze", f"{len(active_df):,}")
	m2.metric("Wyświetlane kolumny", len(selected_columns))
	if TARGET_COL in active_df.columns:
		m3.metric("Mediana zmiennej docelowej", f"{active_df[TARGET_COL].median():.2f}")
	else:
		m3.metric("Mediana zmiennej docelowej", "Brak")

	st.dataframe(view_df.head(row_limit), use_container_width=True)

	numeric_subset = [col for col in selected_columns if pd.api.types.is_numeric_dtype(active_df[col])]
	if numeric_subset:
		st.subheader("Statystyki podsumowujące")
		st.dataframe(active_df[numeric_subset].describe().transpose(), use_container_width=True)


def render_prediction_view(df_clean: pd.DataFrame) -> None:
	st.header("Predykcja ceny")
	st.caption("Predykcja wykonywana przez API na podstawie request body.")

	with st.form("prediction_form"):
		payload = build_prediction_payload(df_clean, key_prefix="predict")
		submitted = st.form_submit_button("Predykcja ceny", type="primary")

	if submitted:
		try:
			result = api.predict_price(payload)
			suggested_price = float(result["suggested_price"])
			currency = result.get("currency", "USD")
			st.success(f"Przewidywana cena za noc: {suggested_price:.2f} {currency}")
			with st.expander("Szczegóły request/response", expanded=False):
				st.json({"request": payload, "response": result})
		except Exception as exc:
			st.error(f"Błąd predykcji API: {exc}")


def main() -> None:
	st.set_page_config(page_title="Airbnb Dashboard", layout="wide")
	st.title("Oferty Airbnb w Nowym Jorku")
	st.caption("Przegląd danych treningowych i predykcja cen noclegów przez API.")

	try:
		df_clean = load_and_prepare_data()
	except FileNotFoundError as exc:
		st.error(str(exc))
		st.stop()

	with st.sidebar:
		st.header("Widoki")
		selected_view = st.radio(
			"Wybierz widok dashboardu",
			options=["Przegląd danych", "Predykcja ceny"],
		)

	if selected_view == "Przegląd danych":
		render_dataset_browser(df_clean)
	else:
		render_prediction_view(df_clean)


if __name__ == "__main__":
	main()
