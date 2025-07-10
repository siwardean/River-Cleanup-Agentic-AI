import os
import json
from ultralytics import YOLO

# ✅ Load YOLOv9 custom model
model = YOLO(r"C:/Users/prasamsha.sharma/River_Model/Best_Model.pt")  # change path if needed

LABELS = ['plastic_bottle', 'Juice box', 'plastic bag', 'other_waste']  # customize based on the model

def run_yolo_inference(image_path):
    results = model(image_path)  # returns a list with one Results object
    detections = results[0].tojson(normalize=True)  # normalized JSON string
    return json.loads(detections)  # parsed as list of dicts

def save_annotated_image(results, save_path="static/annotated_output.jpg"):
    results[0].save(filename=save_path)  # Save annotated image to specified path
    return save_path


def generate_river_description(yolo_detections):
    if not yolo_detections:
        return "No visible pollutants detected in the river."

    labels = [det['name'] for det in yolo_detections if 'name' in det]
    counts = {label: labels.count(label) for label in set(labels)}

    description_parts = [f"{count} {label.replace('_', ' ')}(s)" for label, count in counts.items()]
    return f"The river image shows: {', '.join(description_parts)}."

def load_yolo_prediction(json_path):
    with open(json_path, "r") as f:
        return json.load(f)
