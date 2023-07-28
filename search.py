import requests
import json
from config import SERP_API_KEY

def search(researchObject):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": researchObject})
    headers = {'X-API-KEY': SERP_API_KEY, 'Content-Type': 'application/json'}
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    return response.text
