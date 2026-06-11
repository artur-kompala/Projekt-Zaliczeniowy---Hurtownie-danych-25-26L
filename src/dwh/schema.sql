-- 1. WYMIAR: Czas
CREATE TABLE IF NOT EXISTS dim_date (
    date_key DATE PRIMARY KEY,
    day_of_week INT,
    month_actual INT,
    year_actual INT,
    is_weekend BOOLEAN
);

-- 2. WYMIAR: Host
CREATE TABLE IF NOT EXISTS dim_host (
    host_id BIGINT PRIMARY KEY,
    host_name VARCHAR(255),
    host_since DATE,
    host_is_superhost BOOLEAN
);

-- 3. WYMIAR: Listing
CREATE TABLE IF NOT EXISTS dim_listing (
    listing_id BIGINT PRIMARY KEY,
    name VARCHAR(255),
    room_type VARCHAR(50),
    accommodates INT,
    bathrooms FLOAT,
    bedrooms INT,
    beds INT,
    latitude FLOAT,
    longitude FLOAT,
    host_id BIGINT REFERENCES dim_host(host_id)
);

-- 4. TABELA FAKTÓW: Kalendarz i Ceny
CREATE TABLE IF NOT EXISTS fact_calendar (
    fact_id SERIAL PRIMARY KEY,
    listing_id BIGINT REFERENCES dim_listing(listing_id),
    date_key DATE REFERENCES dim_date(date_key),
    price FLOAT,
    adjusted_price FLOAT,
    minimum_nights INT,
    maximum_nights INT,
    is_available BOOLEAN
);