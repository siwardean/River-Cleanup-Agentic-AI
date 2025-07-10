from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = AzureOpenAI(
     api_key = os.getenv("GPT_API_KEY"),
    azure_endpoint="https://aiphuahhaiopai01.openai.azure.com/",
    api_version="2024-12-01-preview",
)

DEPLOYMENT_NAME = "gpt-4o"

def generate_recommendation(result_dict):
    prompt = f"""
You're an expert in ML model transferability for environmental systems.

A new river was evaluated. Here's the result:
- Most similar river: {result_dict['most_similar_river']}
- Similarity score: {result_dict['similarity_score']}
- Base model accuracy: {result_dict['base_model_accuracy']}

Should we reuse the model, fine-tune it, or retrain it from scratch?
Provide a clear and actionable recommendation for the ML team.
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a machine learning advisor for river systems."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()
