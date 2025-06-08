import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize the camera and MediaPipe Face Mesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Variables for head movement tracking
_, frame = cam.read()
frame_h, frame_w, _ = frame.shape
initial_nose_x = None  # Store initial nose X position (left/right)
initial_nose_y = None  # Store initial nose Y position (up/down)

# Variables for blink detection
blink_start_time = None

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Mirror the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # **Head-Based Cursor & Scrolling using Nose Tip (Landmark[1])**
        nose = landmarks[1]  # Nose tip landmark
        nose_x, nose_y = int(nose.x * frame_w), int(nose.y * frame_h)

        # Set Initial Nose Position
        if initial_nose_x is None:
            initial_nose_x = nose_x
        if initial_nose_y is None:
            initial_nose_y = nose_y

        # **Detect Head Movement for Cursor Control & Scrolling**
        cursor_speed = 10  # Adjust speed for cursor movement
        movement_threshold = 8  # Adjust for sensitivity

        if nose_x < initial_nose_x - movement_threshold:  # Head moved left
            pyautogui.moveRel(-cursor_speed, 0)  # Move Cursor Left
            print("Moving Cursor Left")
        elif nose_x > initial_nose_x + movement_threshold:  # Head moved right
            pyautogui.moveRel(cursor_speed, 0)  # Move Cursor Right
            print("Moving Cursor Right")

        if nose_y < initial_nose_y - movement_threshold:  # Head moved up
            pyautogui.moveRel(0, -cursor_speed)  # Move Cursor Up
            pyautogui.scroll(20)  # Scroll Up Simultaneously
            print("Moving Cursor Up & Scrolling Up")
        elif nose_y > initial_nose_y + movement_threshold:  # Head moved down
            pyautogui.moveRel(0, cursor_speed)  # Move Cursor Down
            pyautogui.scroll(-20)  # Scroll Down Simultaneously
            print("Moving Cursor Down & Scrolling Down")

        # Draw a Circle at Nose Tip for Debugging
        cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)

        # **Eye Blink Detection for Left & Right Click**
        left_eye = [landmarks[145], landmarks[159]]  # Left Eye Landmarks
        right_eye = [landmarks[374], landmarks[386]]  # Right Eye Landmarks

        left_eye_ratio = left_eye[0].y - left_eye[1].y
        right_eye_ratio = right_eye[0].y - right_eye[1].y

        # Detect Left Blink → Left Click
        if left_eye_ratio < 0.004:  # Eye is closed
            if blink_start_time is None:
                blink_start_time = time.time()  # Start tracking blink time
        else:
            if blink_start_time:
                blink_duration = time.time() - blink_start_time
                blink_start_time = None  # Reset timer

                if blink_duration < 0.5:
                    pyautogui.click()  # Left Click
                    print("Left Click")

        # Detect Right Blink → Right Click
        if right_eye_ratio < 0.004:  # Eye is closed
            if blink_start_time is None:
                blink_start_time = time.time()  # Start tracking blink time
        else:
            if blink_start_time:
                blink_duration = time.time() - blink_start_time
                blink_start_time = None  # Reset timer

                if blink_duration < 0.5:
                    pyautogui.rightClick()  # Right Click
                    print("Right Click")

    cv2.imshow('Head & Eye Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cam.release()
cv2.destroyAllWindows()
