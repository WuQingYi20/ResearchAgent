from typing import Any
from bs4 import BeautifulSoup
import base64
import requests
import json

from environment_variables import BROWSERLESS_API_KEY
from summary import summary

def scrape_website(researchObject: str, url: str):
    try:
        auth_header = 'Basic ' + \
            base64.b64encode(BROWSERLESS_API_KEY.encode()).decode()
        print("Scraping website...")
        headers = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
            'Authorization': auth_header,
        }
        data = {
            "url": url
        }
        data_json = json.dumps(data)
        response = requests.post(
            'https://chrome.browserless.io/content',
            headers=headers,
            data=json.dumps(data),
        )
        response.raise_for_status()  # raise exception for 4XX or 5XX responses

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # extract meaningful text using common tags
        text = ' '.join(map(lambda p: p.text, soup.find_all(
            ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article', 'div', 'span'])))

        # Remove extra whitespaces
        text = ' '.join(text.split())

        print("Content: ", text)

        if len(text) > 10000:
            text = summary(researchObject, text)
        return text

    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return None
    except requests.exceptions.RequestException as req_err:
        print(f'Request error occurred: {req_err}')
        return None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None
