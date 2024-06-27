import os
from ultralytics import YOLO

def setup_model():
    # Initialize a new YOLO model with a larger architecture
    model = YOLO('yolov8l.pt')  
    return model

def train_model(model, data_yaml_path):
    # Train the model with augmented dataset and tuned hyperparameters
    results = model.train(
        data=data_yaml_path,
        epochs=500,  # Increase the number of epochs
        imgsz=640,
        batch=32,  # Adjust batch size based on your GPU memory
        name='frc_yolov8l',  
        optimizer='Adam',  # Try using Adam optimizer
        lr0=0.001,  # Initial learning rate
        augment=True  # Enable data augmentation
    )
    return results

def validate_model(model):
    # Validate the model
    val_results = model.val()
    return val_results

def main():
    data_yaml_path = "/home/zerui/FRC-Game-Piece-Detection/2024-Game-Piece-2/data.yaml"
    model = setup_model()
    train_results = train_model(model, data_yaml_path)
    print(train_results)
    val_results = validate_model(model)
    print(val_results)

if __name__ == "__main__":
    main()
