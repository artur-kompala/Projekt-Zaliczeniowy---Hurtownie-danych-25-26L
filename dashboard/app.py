import streamlit as st
import pandas as pd
import requests
from sqlalchemy import create_engine

# Konfiguracja adresów
API_URL = "http://127.0.0.1:8000/predict"
DATABASE_URL = "postgresql://analytics_user:secure_password_123@127.0.0.1:5432/airbnb_dwh"

st.set_page_config(page_title="Airbnb NYC Analytics", layout="wide")

st.title("🗽 System Analityczny Airbnb - Nowy Jork")
st.markdown("Kompletny projekt hurtowni danych i MLOps. Wybierz moduł poniżej:")

# Podział na dwie zakładki
tab1, tab2 = st.tabs(["📊 Hurtownia Danych (PostgreSQL)", "🤖 Sztuczna Inteligencja (FastAPI + MLflow)"])

with tab1:
    st.header("Analiza historyczna z Hurtowni Danych")
    st.markdown("Pobieranie zagregowanych danych bezpośrednio z bazy `airbnb_dwh`.")

    # Przycisk, aby nie odpytywać bazy przy każdym odświeżeniu aplikacji
    if st.button("Pobierz statystyki typów pokoi"):
        try:
            engine = create_engine(DATABASE_URL)
            query = """
                            SELECT 
                                COALESCE(d.room_type, 'Nieznany') AS "Typ Pokoju", 
                                COUNT(f.fact_id) AS "Liczba Rezerwacji",
                                ROUND(AVG(f.price)::numeric, 2) AS "Średnia Cena ($)"
                            FROM fact_calendar f
                            LEFT JOIN dim_listing d ON f.listing_id = d.listing_id
                            WHERE f.price > 0
                            GROUP BY COALESCE(d.room_type, 'Nieznany')
                            ORDER BY "Liczba Rezerwacji" DESC;
                        """
            df_stats = pd.read_sql(query, engine)

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df_stats, use_container_width=True)
            with col2:
                # Prosty wykres wbudowany w Streamlit
                st.bar_chart(data=df_stats.set_index("Typ Pokoju")["Średnia Cena ($)"])

        except Exception as e:
            st.error(f"Błąd połączenia z bazą: {e}")

with tab2:
    st.header("Wycena Nieruchomości")
    st.markdown("Ustaw parametry mieszkania, aby model **LightGBM** oszacował jego rynkową cenę za noc.")

    col1, col2 = st.columns(2)

    with col1:
        room_type = st.selectbox("Typ pokoju", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
        accommodates = st.slider("Liczba gości", 1, 16, 4)
        bedrooms = st.slider("Liczba sypialni", 1, 10, 1)
        beds = st.slider("Liczba łóżek", 1, 15, 2)

    with col2:
        bathrooms = st.number_input("Liczba łazienek", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        minimum_nights = st.number_input("Minimalna liczba nocy", min_value=1, value=2)
        # Przybliżone koordynaty dla Nowego Jorku
        latitude = st.number_input("Szerokość geograficzna (Latitude)", value=40.7128, format="%.4f")
        longitude = st.number_input("Długość geograficzna (Longitude)", value=-74.0060, format="%.4f")

    if st.button("Oblicz sugerowaną cenę", type="primary"):
        # Przygotowanie paczki (JSON) wysyłanej do FastAPI
        payload = {
            "minimum_nights": minimum_nights,
            "accommodates": accommodates,
            "bathrooms": bathrooms,
            "bedrooms": bedrooms,
            "beds": beds,
            "latitude": latitude,
            "longitude": longitude,
            "room_type": room_type
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                price = result['suggested_price']
                st.success(f"### Oszacowana cena: **{price} $** za noc")
                st.balloons()
            else:
                st.error(f"Błąd z API: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Błąd: Nie można połączyć się z FastAPI. Czy serwer Uvicorn działa na porcie 8000?")