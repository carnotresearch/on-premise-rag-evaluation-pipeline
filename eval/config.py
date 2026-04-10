import os
from dotenv import load_dotenv
# from eval.config import API_URL, MODEL_NAME
load_dotenv()

API_URL = os.getenv("API_URL")
MODEL_NAME = os.getenv("MODEL_NAME")