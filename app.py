import os
import streamlit as st
from agents.model_evaluation_agent import ModelEvaluationAgent
from agents.recommendation_agent import generate_recommendation
from agents.vision_agent import (
    run_yolo_inference,
    save_annotated_image,
    generate_river_description
)

# Page setup
st.set_page_config(page_title="River Cleanup Assistant", layout="wide")
st.title("🌍 River Cleanup AI Assistant")

st.markdown(
    """
    Upload an image of a river. The system will detect pollution, describe the environment,
    evaluate ML model transferability, and suggest improvements.
    """
)

# Upload image
uploaded_image = st.file_uploader("📤 Upload River Image", type=["jpg", "png"])

if uploaded_image:
    # Save the image
    image_path = os.path.join("static", "uploaded_river.jpg")
    with open(image_path, "wb") as f:
        f.write(uploaded_image.read())

    # Display uploaded image
    st.image(image_path, caption="📷 Uploaded River Image", use_column_width=True)

    # Run YOLO detection
    st.markdown("### 🔍 Detecting pollution...")
    from agents.vision_agent import model  # 🔁 import model to access YOLO object

    results = model(image_path)
    detections = results[0].to_df().to_dict(orient="records")


# Save + display annotated image
    annotated_path = save_annotated_image(results)
    st.image(annotated_path, caption="🧪 Detected Pollutants", use_column_width=True)


    # Generate river description
    st.markdown("### 📝 River Description")
    river_description = generate_river_description(detections)
    st.write(river_description)

    # Model evaluation
    st.markdown("### 📊 Model Evaluation")
    evaluator = ModelEvaluationAgent()
    # Wrap the river description in a dict as expected by evaluate_new_river
    new_metadata = {
    "caption": river_description,
    "location": "unknown",  # or get actual location if you have it
    "timestamp": "unknown"  # or get actual timestamp if available
}
    evaluation_result = evaluator.evaluate_new_river(new_metadata)

    st.json(evaluation_result)

    # Recommendation
    st.markdown("### 💡 Recommendation")
    recommendation = generate_recommendation(evaluation_result)
    st.write(recommendation)
else:
    st.info("👈 Please upload an image to begin analysis.")
