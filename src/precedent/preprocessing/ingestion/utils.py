import logging

import requests

logger = logging.getLogger(__name__)


class IngestionUtils:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch(self, package_id):
        """
        Fetches the XML data from the specified URL using the provided API key.
        Returns the XML content as a string.
        """
        url = f"https://api.govinfo.gov/packages/{package_id}/xml"
        params = {"api_key": self.api_key}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch package {package_id}: {e}")
            return None
