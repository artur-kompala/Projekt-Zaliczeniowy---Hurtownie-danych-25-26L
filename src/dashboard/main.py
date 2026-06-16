import os
from pathlib import Path

import pandas as pd
import streamlit as st
from api_middleware import ApiMiddleware

API_URL = os.getenv("API_URL", "http://localhost:2137")
api: ApiMiddleware = ApiMiddleware(api_url=API_URL)

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
	records = api.get_listings_data(limit=25).get("records", [])
	return pd.DataFrame(records)

def build_prediction_payload(df_clean: pd.DataFrame, key_prefix: str) -> dict:
	if not isinstance(df_clean, pd.DataFrame):
		df_clean = pd.DataFrame(df_clean)

	st.subheader("Dane wejściowe predykcji")

	st.markdown("#### Parametry liczbowe")
	n1, n2 = st.columns(2)
	numeric_fields = [
		("Minimum nights", "minimum_nights", 1, 365, int(df_clean["minimum_nights"].median()), 1),
		("Accommodates", "accommodates", 1, 16, int(df_clean["accommodates"].median()), 1),
		("Bathrooms", "bathrooms", 0.0, 10.0, float(df_clean["bathrooms"].median()), 0.5),
		("Bedrooms", "bedrooms", 0, 20, int(df_clean["bedrooms"].median()), 1),
		("Host listings count", "host_listings_count", 0, 100, int(df_clean["host_listings_count"].median()), 1),
		("Maximum nights", "maximum_nights", 1, 365, int(df_clean["maximum_nights"].median()), 1),
		("Availability 365", "availability_365", 0, 365, int(df_clean["availability_365"].median()), 1),
		("Number of reviews", "number_of_reviews", 0, 1000, int(df_clean["number_of_reviews"].median()), 1),
		("Number of reviews LTM", "number_of_reviews_ltm", 0, 1000, int(df_clean["number_of_reviews_ltm"].median()), 1),
		("Review scores rating", "review_scores_rating", 0.0, 100.0, float(df_clean["review_scores_rating"].median()), 0.5),
		("Reviews per month", "reviews_per_month", 0.0, 100.0, float(df_clean["reviews_per_month"].median()), 0.1),
	]

	numeric_values = {}
	for index, (label, field_name, minimum_value, maximum_value, default_value, step_value) in enumerate(numeric_fields):
		with n1 if index % 2 == 0 else n2:
			numeric_values[field_name] = st.number_input(
				label,
				min_value=minimum_value,
				max_value=maximum_value,
				value=default_value,
				step=step_value,
				key=f"{key_prefix}_{field_name}",
			)

	st.markdown("#### Parametry kategoryczne")
	categories = api.get_categories()
	c1, c2 = st.columns(2)
	with c1:
		room_type = st.selectbox(
			"Room type",
			options=categories["room_type"],
			index=1,
			key=f"{key_prefix}_room_type",
		)
		property_type = st.selectbox(
			"Property type",
			options=categories["property_type"],
			index=0,
			key=f"{key_prefix}_property_type",
		)
	with c2:
		neighbourhood_group_cleansed = st.selectbox(
			"Neighbourhood group",
			options=categories["neighbourhood_group_cleansed"],
			index=0,
			key=f"{key_prefix}_neighbourhood_group_cleansed",
		)
		neighbourhood_cleansed = st.selectbox(
			"Neighbourhood",
			options=categories["neighbourhood_cleansed"],
			index=0,
			key=f"{key_prefix}_neighbourhood_cleansed",
		)

	return {
		"room_type" : room_type,
		"property_type" : property_type,
		"neighbourhood_group_cleansed" : neighbourhood_group_cleansed,
		"neighbourhood_cleansed" : neighbourhood_cleansed,

		"host_listings_count" : int(numeric_values["host_listings_count"]),
		"accommodates" : int(numeric_values["accommodates"]),
		"bathrooms" : float(numeric_values["bathrooms"]),
		"bedrooms" : int(numeric_values["bedrooms"]),
		"minimum_nights" : int(numeric_values["minimum_nights"]),
		"maximum_nights" : int(numeric_values["maximum_nights"]),
		"availability_365" : int(numeric_values["availability_365"]),
		"number_of_reviews" : int(numeric_values["number_of_reviews"]),
		"number_of_reviews_ltm" : int(numeric_values["number_of_reviews_ltm"]),
		"review_scores_rating" : float(numeric_values["review_scores_rating"]),
		"reviews_per_month" : float(numeric_values["reviews_per_month"])
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
