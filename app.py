from flask import Flask, Response, stream_with_context, render_template
import cv2
import numpy as np
import mediapipe as mp

# Import your existing functions and code
from debug2 import add_logo, gesture_detection  # Assuming debug2.py and debug3.py are in the same directory

app = Flask(__name__)

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load logo
logo_path = 'static/images/logo.png'  # Update this path to your logo file
logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

def generate_frames():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        textt = ''
        check = ''
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Detect gesture
                    gesture = gesture_detection(hand_landmarks.landmark)

                    if check != str(gesture):
                        if str(gesture) != "None":
                            check = str(gesture)
                            textt = textt + " " + str(gesture)

            # Define black background properties
            text_background_height = 50
            frame_height, frame_width, _ = frame.shape

            # Create a black background at the bottom of the frame
            black_background = np.zeros((text_background_height, frame_width, 3), dtype=np.uint8)
            black_background[:] = (0, 0, 0)  # Set to black

            # Copy the original frame into a new frame to overlay the black background
            frame_with_background = frame.copy()
            frame_with_background[-text_background_height:, :] = black_background

            # Calculate text size and position
            text_size, _ = cv2.getTextSize(textt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_width, text_height = text_size
            bottom_left_x = 10
            bottom_left_y = text_background_height - 10  # Position text within the black background

            # Add text on top of the black background
            cv2.putText(frame_with_background, textt, (bottom_left_x, bottom_left_y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            if len(textt) > 30:
                textt = ''

            # Encode frame as JPEG and yield
            _, buffer = cv2.imencode('.jpg', frame_with_background)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(generate_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
