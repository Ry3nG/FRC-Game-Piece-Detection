import os
from ultralytics import YOLO

def setup_model():
    #initialize a new YOLO model
    model = YOLO('yolov8n.pt')
    return model

def train_model(model, data_yaml_path):
    # train the model
    results = model.train(data = data_yaml_path, epochs = 3,imgsz = 640, batch = 16, name = 'frc_yolov8n')
    return results

def validate_model(model):
    #validate the model
    val_results = model.val()
    return val_results

def main():
    data_yaml_path ="/home/zerui/FRC-Game-Piece-Detection/2024-Game-Piece-2/data.yaml"
    model = setup_model()
    train_results = train_model(model, data_yaml_path)
    print(train_results)
    val_results = validate_model(model)
    print(val_results)

if __name__ == "__main__":
    main()