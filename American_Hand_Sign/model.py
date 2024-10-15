import cv2
from ultralytics import YOLO
import torch

# Check if CUDA (GPU) is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained YOLOv11 model
model = YOLO("best.pt").to(device)  

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, or use a video file path

# Set the desired frame size (640x640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Predict on the current frame
    results = model(frame, imgsz=640, show=True)  # imgsz=640 sets the image size

    # Show the result on screen
    cv2.imshow("YOLOv11 Hand Sign Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
