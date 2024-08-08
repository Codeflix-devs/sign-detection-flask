import time
import numpy as np
import cv2
import mediapipe as mp
import datetime

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

logo_path = 'logo.png'  # Update this path to your logo file
logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel


def add_logo(frame, logo):
    # Resize logo to fit the width of the frame
    frame_height, frame_width, _ = frame.shape
    logo_height, logo_width = logo.shape[:2]
    scale = frame_width / 6 / logo_width
    # scale = 1
    new_width = int(logo_width * scale)
    new_height = int(logo_height * scale)

    logo_resized = cv2.resize(logo, (new_width, new_height), interpolation=cv2.INTER_AREA)
    logo_height, logo_width = logo_resized.shape[:2]

    # Create a mask of logo and its inverse mask
    if logo_resized.shape[2] == 4:  # Check if the logo has an alpha channel
        alpha_channel = logo_resized[:, :, 3] / 255.0
        logo_rgb = logo_resized[:, :, :3]
    else:
        alpha_channel = np.ones((logo_height, logo_width))
        logo_rgb = logo_resized

    # Calculate position
    x_offset = (frame_width - logo_width) // 2
    y_offset = 10  # Padding from top

    # Region of interest in the frame
    roi = frame[y_offset:y_offset + logo_height, x_offset:x_offset + logo_width]

    # Blend the logo with the ROI
    for c in range(0, 3):
        roi[:, :, c] = (alpha_channel * logo_rgb[:, :, c] + (1 - alpha_channel) * roi[:, :, c])

    frame[y_offset:y_offset + logo_height, x_offset:x_offset + logo_width] = roi


# Define gesture detection function
def gesture_detection(hand_landmarks):
    # Ensure landmarks are provided
    if not hand_landmarks:
        return "No landmarks detected"

    # Get key points

    thumb_cmc = hand_landmarks[mp_hands.HandLandmark.THUMB_CMC]
    thumb_mcp = hand_landmarks[mp_hands.HandLandmark.THUMB_MCP]
    thumb_ip = hand_landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_tip = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_mcp = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_finger_pip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_finger_dip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_finger_tip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_mcp = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    middle_finger_pip = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_finger_dip = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    middle_finger_tip = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_mcp = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    ring_finger_pip = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    ring_finger_dip = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]
    ring_finger_tip = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks[mp_hands.HandLandmark.PINKY_MCP]
    pinky_pip = hand_landmarks[mp_hands.HandLandmark.PINKY_PIP]
    pinky_dip = hand_landmarks[mp_hands.HandLandmark.PINKY_DIP]



    # print("*************************************************************************")
    # print(f"thumb_cmc: ({thumb_cmc.x}, {thumb_cmc.y})")
    # print(f"thumb mcp: ({thumb_mcp.x}, {thumb_mcp.y})")
    # print(f"thumb ip: ({thumb_ip.x}, {thumb_ip.y})")
    # print(f"thumb Tip: ({thumb_tip.x}, {thumb_tip.y})")
    # print(f"index finger mcp: ({index_finger_mcp.x}, {index_finger_mcp.y})")
    # print(f"index finger pip: ({index_finger_pip.x}, {index_finger_pip.y})")
    # print(f"index finger dip: ({index_finger_dip.x}, {index_finger_dip.y})")
    # print(f"index finger tip: ({index_finger_tip.x}, {index_finger_tip.y})")
    # print(f"middle finger mcp: ({middle_finger_mcp.x}, {middle_finger_mcp.y})")
    # print(f"middle finger pip: ({middle_finger_pip.x}, {middle_finger_pip.y})")
    # print(f"middle finger dip: ({middle_finger_dip.x}, {middle_finger_dip.y})")
    # print(f"middle finger tip: ({middle_finger_tip.x}, {middle_finger_tip.y})")
    # print(f"ring finger mcp: ({ring_finger_mcp.x}, {ring_finger_mcp.y})")
    # print(f"ring finger pip: ({ring_finger_pip.x}, {ring_finger_pip.y})")
    # print(f"ring finger dip: ({ring_finger_dip.x}, {ring_finger_dip.y})")
    # print(f"ring finger tip: ({ring_finger_tip.x}, {ring_finger_tip.y})")
    # print(f"Pinky Tip: ({pinky_tip.x}, {pinky_tip.y})")
    # print(f"Pinky MCP: ({pinky_mcp.x}, {pinky_mcp.y})")
    # print(f"Pinky pip: ({pinky_pip.x}, {pinky_pip.y})")
    # print(f"Pinky diP: ({pinky_dip.x}, {pinky_dip.y})")
    #
    # print("**********************************************************************************")

    # if index_finger_mcp.y > index_finger_pip.y and index_finger_dip.y > index_finger_pip.y and index_finger_tip.y > index_finger_dip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_pip.y > middle_finger_dip.y and middle_finger_dip.y > middle_finger_tip.y:
    #     return "index close"
    # elif middle_finger_mcp.y > middle_finger_pip.y and middle_finger_dip.y > middle_finger_pip.y and middle_finger_tip.y > middle_finger_dip.y and ring_finger_mcp.y > ring_finger_pip.y and ring_finger_pip.y > ring_finger_dip.y and ring_finger_dip.y > ring_finger_tip.y:
    #     return "middle close"
    # elif ring_finger_mcp.y > ring_finger_pip.y and ring_finger_dip.y > ring_finger_pip.y and ring_finger_tip.y > ring_finger_dip.y and pinky_mcp.y > pinky_pip.y and pinky_pip.y > pinky_dip.y and pinky_dip.y > pinky_tip.y:
    #     return "ring close"
    # elif pinky_mcp.y > pinky_pip.y and pinky_dip.y > pinky_pip.y and pinky_tip.y > pinky_dip.y and index_finger_mcp.y > index_finger_pip.y and index_finger_pip.y > index_finger_dip.y and index_finger_dip.y > index_finger_tip.y and ring_finger_mcp.y > ring_finger_pip.y and ring_finger_pip.y > ring_finger_dip.y and ring_finger_dip.y > ring_finger_tip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_pip.y > middle_finger_dip.y and middle_finger_dip.y > middle_finger_tip.y:
    #     return "pinky close"


    # elif thumb_cmc.y > thumb_mcp.y and thumb_mcp.y > thumb_ip.y and thumb_ip.y > thumb_tip.y and index_finger_mcp.y > index_finger_pip.y and index_finger_pip.y > index_finger_dip.y and index_finger_dip.y > index_finger_tip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_pip.y > middle_finger_dip.y and middle_finger_dip.y > middle_finger_tip.y and ring_finger_mcp.y>ring_finger_pip.y and ring_finger_pip.y < ring_finger_dip.y and ring_finger_tip.y > ring_finger_dip.y and ring_finger_tip.y > ring_finger_mcp.y and pinky_mcp.y > pinky_pip.y and pinky_pip.y < pinky_dip.y and pinky_tip.y>pinky_dip.y and pinky_tip.y > pinky_mcp.y:
    #     return "V"

    if thumb_cmc.y > thumb_mcp.y and thumb_mcp.y > thumb_ip.y and thumb_ip.y > thumb_tip.y and index_finger_mcp.y > index_finger_pip.y and index_finger_pip.y > index_finger_dip.y and index_finger_dip.y > index_finger_tip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_dip.y > middle_finger_pip.y and middle_finger_tip.y > middle_finger_dip.y and ring_finger_mcp.y > ring_finger_pip.y and ring_finger_dip.y > ring_finger_pip.y and ring_finger_tip.y > ring_finger_dip.y and pinky_mcp.y > pinky_pip.y and pinky_pip.y > pinky_dip.y and pinky_dip.y > pinky_tip.y:
        return "Love You"
    elif thumb_cmc.x < thumb_mcp.x and thumb_mcp.x > thumb_ip.x and thumb_ip.x > thumb_tip.x and index_finger_mcp.y > index_finger_pip.y and index_finger_dip.y > index_finger_pip.y and index_finger_tip.y > index_finger_dip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_dip.y > middle_finger_pip.y and middle_finger_tip.y > middle_finger_dip.y and ring_finger_mcp.y > ring_finger_pip.y and ring_finger_dip.y > ring_finger_pip.y and ring_finger_tip.y > ring_finger_dip.y and pinky_mcp.y > pinky_pip.y and pinky_dip.y > pinky_pip.y and pinky_tip.y > pinky_dip.y:
        return "Rock"

    # elif thumb_cmc.x < thumb_mcp.x and thumb_mcp.x < thumb_ip.x and thumb_ip.x < thumb_tip.x and index_finger_mcp.y > index_finger_pip.y and index_finger_pip.y > index_finger_dip.y and index_finger_dip.y > index_finger_tip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_dip.y > middle_finger_pip.y and middle_finger_tip.y > middle_finger_dip.y:
    #     return "L"

    elif thumb_cmc.x < thumb_mcp.x and thumb_mcp.x < thumb_ip.x and thumb_ip.x < thumb_tip.x and thumb_cmc.y > thumb_mcp.y and thumb_mcp.y > thumb_ip.y and thumb_ip.y > thumb_tip.y and index_finger_mcp.y > index_finger_pip.y and index_finger_dip.y > index_finger_pip.y and index_finger_tip.y > index_finger_dip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_dip.y > middle_finger_pip.y and middle_finger_tip.y > middle_finger_dip.y and ring_finger_mcp.y > ring_finger_pip.y and ring_finger_dip.y > ring_finger_pip.y and ring_finger_tip.y > ring_finger_dip.y and pinky_mcp.y > pinky_pip.y and pinky_pip.y > pinky_dip.y and pinky_dip.y > pinky_tip.y:
        return "Call Me"
    elif thumb_cmc.y > thumb_mcp.y and thumb_mcp.y > thumb_ip.y and thumb_ip.y > thumb_tip.y and index_finger_mcp.y > index_finger_pip.y and index_finger_pip.y > index_finger_dip.y and index_finger_dip.y > index_finger_tip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_pip.y > middle_finger_dip.y and middle_finger_dip.y > middle_finger_tip.y and ring_finger_mcp.y > ring_finger_pip.y and ring_finger_pip.y > ring_finger_dip.y and ring_finger_dip.y > ring_finger_tip.y and pinky_mcp.y > pinky_pip.y and pinky_dip.y > pinky_pip.y and pinky_tip.y > pinky_dip.y:
        return "Three"

    elif thumb_cmc.x < thumb_mcp.x and thumb_mcp.x < thumb_ip.x and thumb_ip.x < thumb_tip.x and index_finger_mcp.y > index_finger_pip.y and index_finger_dip.y > index_finger_pip.y and index_finger_tip.y > index_finger_dip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_dip.y > middle_finger_pip.y and middle_finger_tip.y > middle_finger_dip.y and ring_finger_mcp.y > ring_finger_pip.y and ring_finger_dip.y > ring_finger_pip.y and ring_finger_tip.y > ring_finger_dip.y and pinky_mcp.y > pinky_pip.y and pinky_dip.y > pinky_pip.y and pinky_tip.y > pinky_dip.y and index_finger_mcp.x < index_finger_pip.x and index_finger_dip.x > index_finger_tip.x:
        return "How are you??"

    elif thumb_cmc.x < thumb_mcp.x and thumb_mcp.x > thumb_ip.x and thumb_ip.x > thumb_tip.x and index_finger_mcp.y > index_finger_pip.y and index_finger_pip.y > index_finger_dip.y and index_finger_dip.y > index_finger_tip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_pip.y > middle_finger_dip.y and middle_finger_dip.y > middle_finger_tip.y and ring_finger_mcp.y > ring_finger_pip.y and ring_finger_pip.y > ring_finger_dip.y and ring_finger_dip.y > ring_finger_tip.y and pinky_mcp.y > pinky_pip.y and pinky_pip.y > pinky_dip.y and pinky_dip.y > pinky_tip.y:
        return "Four"

    elif thumb_cmc.x < thumb_mcp.x and thumb_mcp.x < thumb_ip.x and thumb_ip.x < thumb_tip.x and index_finger_mcp.y > index_finger_pip.y and index_finger_pip.y > index_finger_dip.y and index_finger_dip.y > index_finger_tip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_pip.y > middle_finger_dip.y and middle_finger_dip.y > middle_finger_tip.y and ring_finger_mcp.y > ring_finger_pip.y and ring_finger_pip.y > ring_finger_dip.y and ring_finger_dip.y > ring_finger_tip.y and pinky_mcp.y > pinky_pip.y and pinky_pip.y > pinky_dip.y and pinky_dip.y > pinky_tip.y:
        return "Hii"

    elif thumb_cmc.x < thumb_mcp.x and thumb_mcp.x < thumb_ip.x and thumb_ip.x < thumb_tip.x and index_finger_mcp.y < index_finger_pip.y and index_finger_pip.y < index_finger_dip.y and index_finger_dip.y > index_finger_tip.y and index_finger_mcp.x < index_finger_pip.x and index_finger_pip.x > index_finger_dip.x and index_finger_dip.x > index_finger_tip.x and middle_finger_mcp.x < middle_finger_pip.x and middle_finger_pip.x > middle_finger_dip.x and middle_finger_dip.x > middle_finger_tip.x and pinky_mcp.x < pinky_pip.x and pinky_pip.x > pinky_dip.x and pinky_dip.x > pinky_tip.x:
        return "ok"

    elif thumb_cmc.x < thumb_mcp.x and thumb_mcp.x < thumb_ip.x and thumb_ip.x > thumb_tip.x and index_finger_mcp.y > index_finger_pip.y and index_finger_dip.y > index_finger_pip.y and index_finger_tip.y > index_finger_dip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_dip.y > middle_finger_pip.y and middle_finger_tip.y > middle_finger_dip.y and ring_finger_mcp.y > ring_finger_pip.y and ring_finger_dip.y > ring_finger_pip.y and ring_finger_tip.y > ring_finger_dip.y and pinky_mcp.y > pinky_pip.y and pinky_pip.y > pinky_dip.y and pinky_dip.y > pinky_tip.y:
        return "May I GO TO Washroom"
    elif thumb_cmc.x < thumb_mcp.x and thumb_mcp.x < thumb_ip.x and thumb_ip.x < thumb_tip.x and thumb_cmc.y > thumb_mcp.y and thumb_mcp.y > thumb_ip.y and thumb_ip.y > thumb_tip.y and index_finger_mcp.x < index_finger_pip.x and index_finger_pip.x < index_finger_dip.x and index_finger_dip.x < index_finger_tip.x and index_finger_mcp.y > index_finger_pip.y and index_finger_pip.y < index_finger_dip.y and index_finger_dip.y < index_finger_tip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_pip.y > middle_finger_dip.y:
        return "Good"
    elif thumb_cmc.x < thumb_mcp.x and thumb_mcp.x > thumb_ip.x and thumb_ip.x > thumb_tip.x and thumb_cmc.y > thumb_mcp.y and thumb_mcp.y > thumb_ip.y and thumb_ip.y > thumb_tip.y and index_finger_mcp.y > index_finger_pip.y and index_finger_pip.y > index_finger_dip.y and index_finger_dip.y > index_finger_tip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_pip.y > middle_finger_dip.y and middle_finger_dip.y > middle_finger_tip.y and ring_finger_mcp.y > ring_finger_pip.y and ring_finger_pip.y < ring_finger_dip.y and ring_finger_tip.y > ring_finger_dip.y and ring_finger_tip.y > ring_finger_mcp.y and pinky_mcp.y > pinky_pip.y and pinky_pip.y < pinky_dip.y and pinky_tip.y > pinky_dip.y and pinky_tip.y > pinky_mcp.y and middle_finger_mcp.x < middle_finger_pip.x and middle_finger_pip.x < middle_finger_dip.x and middle_finger_dip.x < middle_finger_tip.x:
        return "Promise"
    elif thumb_cmc.x > thumb_mcp.x and thumb_mcp.x > thumb_ip.x and thumb_ip.x > thumb_tip.x and thumb_cmc.y > thumb_mcp.y and thumb_mcp.y > thumb_ip.y and thumb_ip.y > thumb_tip.y and index_finger_mcp.x < index_finger_pip.x and index_finger_pip.x < index_finger_dip.x and index_finger_dip.x < index_finger_tip.x and index_finger_mcp.y > index_finger_pip.y and index_finger_pip.y > index_finger_dip.y and index_finger_dip.y > index_finger_tip.y and middle_finger_mcp.x < middle_finger_pip.x and middle_finger_dip.x > middle_finger_tip.x and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_pip.y < middle_finger_dip.y:
        return "Come Here"
    elif thumb_cmc.x < thumb_mcp.x and thumb_mcp.x > thumb_ip.x and thumb_ip.x > thumb_tip.x and thumb_cmc.y > thumb_mcp.y and thumb_mcp.y > thumb_ip.y and thumb_ip.y > thumb_tip.y and index_finger_mcp.y > index_finger_pip.y and index_finger_pip.y > index_finger_dip.y and index_finger_dip.y > index_finger_tip.y and middle_finger_mcp.y > middle_finger_pip.y and middle_finger_pip.y > middle_finger_dip.y and middle_finger_dip.y > middle_finger_tip.y and ring_finger_mcp.y > ring_finger_pip.y and ring_finger_pip.y < ring_finger_dip.y and ring_finger_tip.y > ring_finger_dip.y and ring_finger_tip.y > ring_finger_mcp.y and pinky_mcp.y > pinky_pip.y and pinky_pip.y < pinky_dip.y and pinky_tip.y > pinky_dip.y and pinky_tip.y > pinky_mcp.y:
        diff = index_finger_tip.x - middle_finger_tip.x
        if diff > 0.05:
            return "Victory"


















global gesture
def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Hand Gesture Recognition', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Hand Gesture Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        textt = ''
        check =''
        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                # print("Failed to grab frame")
                break

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
                            textt = textt+ " " + str(gesture)
                            # if len(textt) > 30:
                            #     # Split text into words
                            #     words = textt.strip().split()
                            #     # Remove words from the start until length is <= 30
                            #     while len(' '.join(words)) > 30:
                            #         words.pop(0)
                            #     # Append the latest gesture
                            #     words.append(str(gesture))
                            #     # Join words to form the new text
                            #     textt = ' '.join(words)
                    # print("chech : ",check)
                    # print("gesture : ",str(gesture))
                    # cv2.putText(frame, textt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # cv2.putText(frame, textt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            text_size, _ = cv2.getTextSize(textt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_width, text_height = text_size

            # Calculate position for text at the bottom
            frame_height, frame_width, _ = frame.shape
            bottom_left_x = 10
            bottom_left_y = frame_height - 50

            add_logo(frame, logo)

            cv2.putText(frame, textt, (bottom_left_x, bottom_left_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if len(textt)>30:


                textt = ''

            cv2.imshow('Hand Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
