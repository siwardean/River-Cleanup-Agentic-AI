from ultralytics import YOLO

def run_predict(model_path='Drone_Model.pt'):
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Predict using the model
    results = model.predict(source=0, save=True, conf=0.37, show=True)
    
    return results
