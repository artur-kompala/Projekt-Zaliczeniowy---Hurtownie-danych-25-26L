import requests


class ApiMiddleware:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")

    def _validate_json_response(self, response: requests.Response, error_prefix: str) -> dict:
        if response.status_code >= 400:
            raise Exception(f"{error_prefix} Kod: {response.status_code}, Treść: {response.text}")

        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise Exception(
                f"{error_prefix} Odpowiedź z API nie jest JSON (Content-Type: {content_type})"
            )

        return response.json()

    def get_listings_data(self, offset: int = 0, limit: int | None = None) -> dict:
        """
        Return value:
        {
          "count": int,
          "limit": int,
          "records": list[dict]
        }
        """
        params = {"offset": offset, "limit": int(limit)} if limit is not None else {"offset": offset}
        response = requests.get(f"{self.api_url}/data/listings", params=params, timeout=20)
        return self._validate_json_response(response, "Nie udało się pobrać danych o ofertach!")

    def get_geo_data(self) -> dict:
        """
        Return value:
        {
          "message": str,
          "artifact_uri": str
        }
        """
        response = requests.get(f"{self.api_url}/data/listings/geo", timeout=15)
        return self._validate_json_response(response, "Nie udało się pobrać danych geograficznych!")

    def get_current_model_info(self) -> dict:
        """
        Return value (example):
        {
          "model_id": str,
          ...other model diagnostics
        }
        """
        response = requests.get(f"{self.api_url}/model/info", timeout=15)
        return self._validate_json_response(response, "Nie udało się pobrać informacji o modelu!")

    def predict_price(self, payload: dict) -> dict:
        """
        payload expected:
        {
          "minimum_nights": int,
          "accommodates": int,
          "bathrooms": float,
          "bedrooms": int,
          "beds": int,
          "latitude": float,
          "longitude": float,
          "room_type": str
        }

        Return value expected:
        {
          "suggested_price": float,
          "currency": str
        }
        """
        response = requests.post(f"{self.api_url}/predict", json=payload, timeout=20)
        return self._validate_json_response(response, "Nie udało się wykonać predykcji ceny!")