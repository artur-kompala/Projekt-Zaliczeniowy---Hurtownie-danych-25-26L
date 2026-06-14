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


def build_safe_feature_name_map(columns: list[str]) -> dict[str, str]:
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


@st.cache_resource
def train_model(
	df_clean: pd.DataFrame,
	n_estimators: int,
	learning_rate: float,
	num_leaves: int,
	max_depth: int,
	min_child_samples: int,
	subsample: float,
	colsample_bytree: float,
	random_state: int,
):
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


def build_input_row(
	df_clean: pd.DataFrame,
	model_columns: list[str],
	feature_name_map: dict[str, str],
	key_prefix: str,
) -> pd.DataFrame:
	input_values: dict[str, object] = {}

	st.subheader("Cechy oferty")
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
				key=f"{key_prefix}_{col}",
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
				key=f"{key_prefix}_{col}",
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
				key=f"{key_prefix}_{col}",
			)

	input_df = pd.DataFrame([input_values])
	input_df = pd.get_dummies(input_df, columns=CATEGORICAL_COLS, dtype=int)
	input_df = input_df.reindex(columns=model_columns, fill_value=0)
	input_df = input_df.rename(columns=feature_name_map)

	return input_df


def get_default_browse_columns() -> list[str]:
	return [TARGET_COL, *NUMERIC_COLS[:4], *CATEGORICAL_COLS[:2]]


def render_dataset_browser(df_clean: pd.DataFrame) -> None:
	st.header("Przegląd danych")
	st.caption("Przeglądaj wybrane kolumny z przygotowanego zbioru danych.")

	all_columns = df_clean.columns.tolist()
	default_columns = [col for col in get_default_browse_columns() if col in all_columns]

	selected_columns = st.multiselect(
		"Kolumny do wyświetlenia",
		options=all_columns,
		default=default_columns,
	)
	if not selected_columns:
		st.warning("Wybierz przynajmniej jedną kolumnę do przeglądania zbioru danych.")
		return

	row_limit = st.slider("Liczba wyświetlanych wierszy", min_value=10, max_value=200, value=25, step=5)
	sort_column = st.selectbox("Sortuj według", options=selected_columns, index=0)
	ascending = st.toggle("Uporządkuj rosnąco", value=True)

	view_df = df_clean[selected_columns].sort_values(by=sort_column, ascending=ascending)

	m1, m2, m3 = st.columns(3)
	m1.metric("Wiersze", f"{len(df_clean):,}")
	m2.metric("Wyświetlane kolumny", len(selected_columns))
	m3.metric("Mediana zmiennej docelowej", f"{df_clean[TARGET_COL].median():.2f}")

	st.dataframe(view_df.head(row_limit), use_container_width=True)

	numeric_subset = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df_clean[col])]
	if numeric_subset:
		st.subheader("Statystyki podsumowujące")
		st.dataframe(df_clean[numeric_subset].describe().transpose(), use_container_width=True)


def render_prediction_view(df_clean: pd.DataFrame) -> None:
	st.header("Predykcja ceny")
	st.caption("Przewiduj cenę noclegu na Airbnb przy użyciu aktualnie załadowanego modelu.")

	model_bundle = st.session_state["model_bundle"]
	metrics = model_bundle["metrics"]
	params = model_bundle["params"]

	m1, m2, m3 = st.columns(3)
	m1.metric("Wiersze użyte", f"{len(df_clean):,}")
	m2.metric("MAE na zbiorze walidacyjnym", f"{metrics['mae']:.2f}")
	m3.metric("R2 na zbiorze walidacyjnym", f"{metrics['r2']:.3f}")

	with st.expander("Aktualne hiperparametry modelu", expanded=False):
		st.json(params)

	with st.form("prediction_form"):
		input_df = build_input_row(
			df_clean,
			model_bundle["model_columns"],
			model_bundle["feature_name_map"],
			key_prefix="predict",
		)
		submitted = st.form_submit_button("Predykcja ceny", type="primary")

	if submitted:
		prediction = float(model_bundle["model"].predict(input_df)[0])
		st.success(f"Przewidywana cena za noc: {prediction:.2f}")


def render_retrain_view(df_clean: pd.DataFrame) -> None:
	st.header("Retraining modelu")
	st.caption("Dostosuj hiperparametry LightGBM i wytrenuj nowy model na przygotowanym zbiorze danych.")

	current_params = st.session_state["model_bundle"]["params"]

	with st.form("retrain_form"):
		left_col, right_col = st.columns(2)

		with left_col:
			n_estimators = st.slider(
				"Liczba estymatorów",
				min_value=50,
				max_value=1000,
				value=int(current_params["n_estimators"]),
				step=50,
			)
			learning_rate = st.slider(
				"Współczynnik uczenia",
				min_value=0.01,
				max_value=0.30,
				value=float(current_params["learning_rate"]),
				step=0.01,
			)
			num_leaves = st.slider(
				"Liczba liści",
				min_value=10,
				max_value=256,
				value=int(current_params["num_leaves"]),
				step=1,
			)
			max_depth = st.slider(
				"Maksymalna głębokość (-1 oznacza brak limitu)",
				min_value=-1,
				max_value=32,
				value=int(current_params["max_depth"]),
				step=1,
			)

		with right_col:
			min_child_samples = st.slider(
				"Min child samples",
				min_value=5,
				max_value=100,
				value=int(current_params["min_child_samples"]),
				step=1,
			)
			subsample = st.slider(
				"Subsample",
				min_value=0.5,
				max_value=1.0,
				value=float(current_params["subsample"]),
				step=0.05,
			)
			colsample_bytree = st.slider(
				"Column sample by tree",
				min_value=0.5,
				max_value=1.0,
				value=float(current_params["colsample_bytree"]),
				step=0.05,
			)
			random_state = st.number_input(
				"Random state",
				min_value=0,
				max_value=9999,
				value=int(current_params["random_state"]),
				step=1,
			)

		submitted = st.form_submit_button("Trenuj model", type="primary")

	if submitted:
		with st.spinner("Trening modelu..."):
			st.session_state["model_bundle"] = train_model(
				df_clean,
				n_estimators=n_estimators,
				learning_rate=learning_rate,
				num_leaves=num_leaves,
				max_depth=max_depth,
				min_child_samples=min_child_samples,
				subsample=subsample,
				colsample_bytree=colsample_bytree,
				random_state=int(random_state),
			)
		st.success("Model został ponownie wytrenowany i załadowany do widoku predykcji.")

	model_bundle = st.session_state["model_bundle"]
	metrics = model_bundle["metrics"]
	m1, m2, m3 = st.columns(3)
	m1.metric("MAE na zbiorze walidacyjnym", f"{metrics['mae']:.2f}")
	m2.metric("R2 na zbiorze walidacyjnym", f"{metrics['r2']:.3f}")
	m3.metric("Liczba cech", len(model_bundle["model_columns"]))

	st.subheader("Najważniejsze cechy modelu")
	st.dataframe(model_bundle["feature_importance"].head(15), use_container_width=True)


def ensure_model_bundle(df_clean: pd.DataFrame) -> None:
	if "model_bundle" not in st.session_state:
		with st.spinner("Trening domyślnego modelu..."):
			st.session_state["model_bundle"] = train_model(df_clean, **DEFAULT_MODEL_PARAMS)


def main() -> None:
	st.set_page_config(page_title="Airbnb Dashboard", layout="wide")
	st.title("Oferty Airbnb w Nowym Jorku")
	st.caption("Przegląd danych treningowych, predykcja cen noclegów i ponowne trenowanie modelu w jednym miejscu.")

	try:
		df_clean = load_and_prepare_data()
	except FileNotFoundError as exc:
		st.error(str(exc))
		st.stop()

	ensure_model_bundle(df_clean)

	with st.sidebar:
		st.header("Widoki")
		selected_view = st.radio(
			"Wybierz widok dashboardu",
			options=["Przegląd danych", "Predykcja ceny", "Ponowne trenowanie modelu"],
		)

	if selected_view == "Przegląd danych":
		render_dataset_browser(df_clean)
	elif selected_view == "Predykcja ceny":
		render_prediction_view(df_clean)
	else:
		render_retrain_view(df_clean)


if __name__ == "__main__":
	main()
