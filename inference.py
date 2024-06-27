import cv2
from ultralytics import YOLO

def process_video(input_path, output_path, model_path, conf_threshold=0.25):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process the video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Draw the bounding boxes on the frame with confidence score filtering
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            for box, confidence in zip(boxes, confidences):
                if confidence >= conf_threshold:
                    x1, y1, x2, y2 = box[:4]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_path = "/home/zerui/FRC-Game-Piece-Detection/video/input/FRC 9483 Robot POV - 2024 Milstein Q67.mp4"
    output_path = "/home/zerui/FRC-Game-Piece-Detection/video/output/output.mp4"
    model_path = "/home/zerui/FRC-Game-Piece-Detection/runs/detect/frc_yolov8l/weights/best.pt"

    print("Starting ... ")
    process_video(input_path, output_path, model_path)
    print("Finished")
