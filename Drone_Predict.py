from ultralytics import YOLO
model = YOLO('Drone_Model.pt')
model.predict(source=0, save=True, conf = 0.37, show = True)