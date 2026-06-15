import requests

class ApiMiddleware:
    def __init__(self, app, api_url):
        self.app = app
        self.api_url = api_url

    async def get_listings_data(self):
        response = requests.get(f"{self.api_url}/listings")
        if response.status_code != 200:
            raise Exception(f"Nie udało się pobrać danych o ofertach! Kod: {response.status_code}")
        return response.json()
    
    async def get_geo_data(self):
        response = requests.get(f"{self.api_url}/geo")
        if response.status_code != 200:
            raise Exception(f"Nie udało się pobrać danych geograficznych! Kod: {response.status_code}")
        return response.json()