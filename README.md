## Wymagania systemowe i uruchomienie

Aby uruchomić projekt na czystym środowisku, należy wykonać poniższe kroki konfiguracji.

### 1. Przygotowanie środowiska wirtualnego Python
Wymagane jest posiadanie zainstalowanego środowiska Python (zalecany 3.11+). W katalogu głównym projektu uruchom:

# Instalacja wymaganych bibliotek
```bash
pip install pandas sqlalchemy psycopg2-binary
```
### 3. Uruchomienie potoku ETL
Wystarczy uruchomić poniższe polecenia, aby system sam pobrał dane i w pełni przygotował hurtownię danych:

```bash
# Krok 1: Automatyczne pobranie surowych danych z Inside Airbnb NYC
python src/etl/download_data.py

# Krok 2: Automatyczne stworzenie tabel (DDL) oraz transformacja i załadowanie danych
python src/etl/load_to_dwh.py

pip install mlflow==2.10.2

pip install fastapi uvicorn pydantic

pip install streamlit requests

 uvicorn src.api.main:app --reload
```

# Zaktualizowane uruchamianie projektu

Projekt zawiera plik zmiennych środowiskowych `.env.default`. Aby uruchomić projekt, skopiuj plik `.env.default` do `.env`.
```console
cp .env.default .env
```

Dodatkowo projekt rozwiązuje pliki requirements dynamicznie, czyli najpierw utwórz środowisko `.venv` modułem venv, a następnie zainstaluj do niego wymagane pakiety. Pamiętaj żeby wrócić do głównego katalogu projektu!

```console
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Następnie uruchom skrypt rozwiązywania zależności:
```console
python scripts/generate_requirements.txt
```

## Docker
W celu uruchomienia dockera wykonaj te kroki:
```console
cd docker
docker compose up
```
Polecenie docker compose up, zarówno zbuduje obrazy dockera jak i uruchomi aplikacje. Obecnie w obrazach dockera jest wszystko poza API i dashboardem. Aby uruchomić API, przejdź do katalogu `src/api` i wykonaj polecenie
```console
python main.py
```
