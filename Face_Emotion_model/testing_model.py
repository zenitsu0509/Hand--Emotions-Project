import cv2
import mediapipe as mp
import pickle
import joblib
import pandas as pd
import numpy as np

# Initialize Mediapipe models
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define landmarks list
landmarks = ['class']
num_coords = 501
for val in range(1, num_coords + 1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

# Load the trained model
model = joblib.load('pose_save_model.pkl')  # Ensure correct model filename

# Start video capture
cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor feed to RGB for Mediapipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detections
        results = holistic.process(image)
        
        # Recolor image back to BGR for OpenCV rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks if available
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Extract pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
        else:
            pose_row = [0] * (num_coords * 4)  # Fallback if no pose landmarks

        # Extract face landmarks if available
        if results.face_landmarks:
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
        else:
            face_row = [0] * (num_coords * 4)  # Fallback if no face landmarks
            
        # Concatenate pose and face rows
        row = pose_row + face_row

        # Make predictions
        try:
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]

            print("Predicted Class:", body_language_class)
            print("Predicted Probabilities:", body_language_prob)
            
            # Get left ear coordinates for placing the label
            coords = tuple(np.multiply(
                np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                          results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                [640, 480]).astype(int))
            
            # Draw rectangle and text for class label
            cv2.rectangle(image, (coords[0], coords[1] + 5), 
                          (coords[0] + len(body_language_class) * 20, coords[1] - 30), 
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw status box
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            
            # Display class
            cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (90, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display probability
            cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), 
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        except Exception as e:
            print(f"Error in prediction: {e}")
        
        # Show webcam feed
        cv2.imshow('Raw Webcam Feed', image)

        # Break loop with 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
