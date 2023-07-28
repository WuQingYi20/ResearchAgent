from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_key = os.getenv('OPENAI_API_KEY')
SERP_API_KEY = os.getenv('SERP_API_KEY')
BROWSERLESS_API_KEY = os.getenv('BROWSERLESS_API_KEY')
