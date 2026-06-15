from sqlalchemy import create_engine


class DbMiddleware:
    """Klasa do zarządzania połączeniem z bazą danych PostgreSQL, używana w różnych częściach projektu do pobierania danych i logowania modeli."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None

    def connect(self):
        """Nawiązuje połączenie z bazą danych i tworzy silnik SQLAlchemy."""
        if self.engine is None:
            self.engine = create_engine(self.database_url)
            print("✅ Połączono z bazą danych.")
        else:
            print("ℹ️ Połączenie z bazą danych już istnieje.")

    def fetch_raw_data(self):
        """Pobiera dane z bazy danych."""
        query = """
            SELECT *
            FROM raw_data
            LIMIT 100000;
        """
        if self.engine is None:
            raise ConnectionError("Nie nawiązano połączenia z bazą danych. Wywołaj metodę connect() przed fetch_raw_data().")
        with self.engine.connect() as connection:
            result = connection.execute(query)
            data = result.fetchall()
            print(f"✅ Pobrano {len(data)} wierszy danych.")
            return data