from openai import AzureOpenAI
import numpy as np
import logging
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)

client = AzureOpenAI(
    api_key = os.getenv("EMBEDDING_API_KEY"),
    azure_endpoint="https://aiphuahhaiopai01.openai.azure.com/",
    api_version="2024-12-01-preview",
)

DEPLOYMENT_NAME = "text-embedding-3-large"

def get_embedding(text: str) -> np.ndarray:
    try:
        response = client.embeddings.create(
            input=[text],
            model=DEPLOYMENT_NAME
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        logging.error(f"Embedding failed: {e}")
        return np.zeros(3072, dtype=np.float32)
