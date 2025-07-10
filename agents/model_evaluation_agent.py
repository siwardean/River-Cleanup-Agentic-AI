import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from agents.embedding_agent import get_embedding

# Sample stored river data
RIVER_ENTRIES = [
    {"river_name": "River A", "description": "Brussel river, plastic pollution, July 2023"},
    {"river_name": "River B", "description": "Luik river, industrial waste, March 2024"},
    {"river_name": "River C", "description": "Gent river, muddy water, natural erosion"},
]

class ModelEvaluationAgent:
    def evaluate_new_river(self, new_metadata):
        query_text = f"{new_metadata['caption']} | {new_metadata['location']} | {new_metadata['timestamp']}"
        query_embedding = get_embedding(query_text)

        similarities = []
        for river in RIVER_ENTRIES:
            emb = get_embedding(river["description"])
            score = cosine_similarity([query_embedding], [emb])[0][0]
            similarities.append((river["river_name"], river["description"], score))

        most_similar = max(similarities, key=lambda x: x[2])

        return {
            "most_similar_river": most_similar[0],
            "similarity_score": round(most_similar[2], 2),
            "base_model_accuracy": 0.71,  # Placeholder for actual eval
            "recommendation": f"⚠️ Suggest fine-tuning: {most_similar[0]} is only sim={round(most_similar[2],2)}"
        }
