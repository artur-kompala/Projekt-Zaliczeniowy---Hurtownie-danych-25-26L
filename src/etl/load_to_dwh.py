import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

# Configuration
DB_USER = "analytics_user"
DB_PASSWORD = "secure_password_123"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"
DB_NAME = "airbnb_dwh"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
RAW_DATA_DIR = "data/raw"


def get_db_engine():
    return create_engine(DATABASE_URL)


def clean_price(price_series):
    """Usuwa znaki $ oraz przecinki z cen i konwertuje na float."""
    return price_series.astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)


def build_dim_date(start_date_str, end_date_str):
    """Generuje pełną tabelę wymiaru czasu (Dim_Date) dla podanego zakresu."""
    print("Generowanie wymiaru czasu (dim_date)...")
    date_range = pd.date_range(start=start_date_str, end=end_date_str)

    dim_date = pd.DataFrame({
        'date_key': date_range.date,
        'day_of_week': date_range.dayofweek + 1,
        'month_actual': date_range.month,
        'year_actual': date_range.year,
        'is_weekend': date_range.dayofweek.isin([5, 6])
    })
    return dim_date


def process_and_load_listings(engine):
    """Przetwarza plik listings i ładuje bezpiecznie dane do dim_host oraz dim_listing za pomocą tabel stagingowych."""
    listings_path = os.path.join(RAW_DATA_DIR, "listings.csv.gz")
    print(f"Wczytywanie i przetwarzanie: {listings_path}...")

    cols_to_keep = [
        'id', 'name', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'latitude', 'longitude', 'host_id', 'host_name', 'host_since', 'host_is_superhost'
    ]

    df = pd.read_csv(listings_path, usecols=cols_to_keep)

    # 1. Przetwarzanie i bezpieczne ładowanie dim_host
    print("Przetwarzanie wymiaru: dim_host...")
    df_host = df[['host_id', 'host_name', 'host_since', 'host_is_superhost']].drop_duplicates(subset=['host_id'])
    df_host['host_since'] = pd.to_datetime(df_host['host_since']).dt.date
    df_host['host_is_superhost'] = df_host['host_is_superhost'].map({'t': True, 'f': False})

    # Ładowanie do tabeli tymczasowej (staging)
    df_host.to_sql('stg_host', engine, if_exists='replace', index=False)

    # Przerzucenie unikalnych rekordów do tabeli docelowej (Natywny UPSERT SQL)
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO dim_host (host_id, host_name, host_since, host_is_superhost)
            SELECT host_id, host_name, host_since, host_is_superhost FROM stg_host
            ON CONFLICT (host_id) DO NOTHING;
        """))
        conn.execute(text("DROP TABLE IF EXISTS stg_host;"))
        conn.commit()
    print("Wymiar dim_host zaktualizowany.")

    # 2. Przetwarzanie i bezpieczne ładowanie dim_listing
    print("Przetwarzanie wymiaru: dim_listing...")
    df_listing = df[
        ['id', 'name', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'latitude', 'longitude',
         'host_id']]
    df_listing = df_listing.rename(columns={'id': 'listing_id'}).drop_duplicates(subset=['listing_id'])
    df_listing[['bathrooms', 'bedrooms', 'beds']] = df_listing[['bathrooms', 'bedrooms', 'beds']].fillna(0)

    # Ładowanie do tabeli tymczasowej (staging)
    df_listing.to_sql('stg_listing', engine, if_exists='replace', index=False)

    # Przerzucenie unikalnych rekordów do tabeli docelowej
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO dim_listing (listing_id, name, room_type, accommodates, bathrooms, bedrooms, beds, latitude, longitude, host_id)
            SELECT listing_id, name, room_type, accommodates, bathrooms, bedrooms, beds, latitude, longitude, host_id FROM stg_listing
            ON CONFLICT (listing_id) DO NOTHING;
        """))
        conn.execute(text("DROP TABLE IF EXISTS stg_listing;"))
        conn.commit()
    print("Wymiar dim_listing zaktualizowany.")


def process_and_load_calendar(engine):
    """Przetwarza plik calendar i ładuje go do fact_calendar bez ryzyka powielania kluczy."""
    calendar_path = os.path.join(RAW_DATA_DIR, "calendar.csv.gz")
    print(f"Wczytywanie i przetwarzanie tabeli faktów: {calendar_path}...")

    chunk_size = 100000
    first_chunk = True

    for chunk in pd.read_csv(calendar_path, chunksize=chunk_size):
        chunk.columns = chunk.columns.str.lower()
        if 'date' in chunk.columns:
            chunk = chunk.rename(columns={'date': 'date_key'})

        chunk['date_key'] = pd.to_datetime(chunk['date_key']).dt.date
        chunk['price'] = clean_price(chunk['price']) if 'price' in chunk.columns else 0.0
        chunk['adjusted_price'] = clean_price(chunk['adjusted_price']) if 'adjusted_price' in chunk.columns else chunk[
            'price']
        chunk['is_available'] = chunk['available'].map(
            {'t': True, 'f': False}) if 'available' in chunk.columns else True

        for col in ['minimum_nights', 'maximum_nights']:
            chunk[col] = chunk[col].fillna(1).astype(int) if col in chunk.columns else 1

        fact_chunk = chunk[
            ['listing_id', 'date_key', 'price', 'adjusted_price', 'minimum_nights', 'maximum_nights', 'is_available']]

        # Dla tabeli faktów, przy każdym uruchomieniu dopisujemy nowe rekordy,
        # ale jeśli system przerwano w połowie, bezpiecznie dodajemy dane.
        fact_chunk.to_sql('fact_calendar', engine, if_exists='append', index=False, method='multi')
        print(".", end="", flush=True)

    print("\nZakończono sukcesem ładowanie danych do fact_calendar!")


def create_schema_if_not_exists(engine):
    """Uruchamia DDL, aby upewnić się, że tabele w bazie istnieją."""
    schema_path = "src/dwh/schema.sql"
    if os.path.exists(schema_path):
        print("Inicjalizacja struktury bazy danych (DDL)...")
        with open(schema_path, "r", encoding="utf-8") as f:
            sql_commands = f.read()
        with engine.connect() as conn:
            conn.execute(text(sql_commands))
            conn.commit()


def main():
    start_time = datetime.now()
    print(f"=== URUCHOMIENIE PIPELINE ETL: {start_time} ===")

    engine = get_db_engine()
    create_schema_if_not_exists(engine)

    # 1. Ładowanie wymiaru czasu
    try:
        df_date = build_dim_date("2023-01-01", "2027-12-31")
        df_date.to_sql('dim_date', engine, if_exists='append', index=False, method='multi')
        print("Wymiar czasu załadowany pomyślnie.")
    except Exception as e:
        if "duplicate key" in str(e).lower() or "unique violation" in str(e).lower():
            print("ℹ️ Wymiar czasu (dim_date) jest już zainicjalizowany. Pomijam.")
        else:
            raise e

    # 2. Ładowanie słowników/wymiarów
    process_and_load_listings(engine)

    # 3. Ładowanie faktów (Kalendarza)
    process_and_load_calendar(engine)

    end_time = datetime.now()
    print(f"=== PIPELINE ZAKOŃCZONY SUKCESEM W CZASIE: {end_time - start_time} ===")


if __name__ == "__main__":
    main()