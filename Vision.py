import cv2
import mediapipe as mp
import numpy as np
import math
import os
import firebase_admin
from firebase_admin import credentials, firestore
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import deque
import base64

# Initialize firebase admin SDK
cred = credentials.Certificate('vision-firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

@dataclass
class GestureConfig:
    COOLDOWN_FRAMES: int = 15
    SWIPE_THRESHOLD: float = 0.15
    SWIPE_FRAMES: int = 5
    MIN_SCALE: float = 0.5
    MAX_SCALE: float = 3.0
    TARGET_SIZE: Tuple[int, int] = (800, 600)

class HandGestureDetector:
    def __init__(self):  # constructor name
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.hand_history = deque(maxlen=5)
        self.circle_detection_history = deque(maxlen=5)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)

    def draw_landmarks(self, frame, hand_landmarks):
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def detect_swipe(self, hand_landmarks) -> Optional[str]:
        palm_x = hand_landmarks.landmark[9].x
        self.hand_history.append(palm_x)
        if len(self.hand_history) == 5:
            total_movement = self.hand_history[-1] - self.hand_history[0]
            if abs(total_movement) > GestureConfig.SWIPE_THRESHOLD:
                return "right" if total_movement < 0 else "left"
        return None

    def detect_circle_gesture(self, hand_landmarks) -> bool:
        # Track the previous positions of the landmarks
        landmarks = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]  # Thumb, Index, Middle, Ring, Pinky
        current_positions = [(lm.x, lm.y) for lm in landmarks]
        self.circle_detection_history.append(current_positions)

        if len(self.circle_detection_history) < 5:
            return False  # Not enough data for circular gesture

        # Compare previous landmark positions to check if they form a circle
        for i in range(1, len(self.circle_detection_history)):
            distance = np.linalg.norm(np.array(self.circle_detection_history[i-1]) - np.array(self.circle_detection_history[i]))
            if distance < 0.05:  # Small distance between consecutive frames means the hand is moving in a circle
                continue
            return False  # If the movement is not circular

        return True  # The gesture seems circular

class ImageController:
    def __init__(self, image_directory: str, config: GestureConfig):
        self.config = config
        self.image_files = self._load_image_files(image_directory)
        self.current_index = 0
        self.scale = 1.0
        self.current_image = None
        self.original_image = None
        self._load_current_image()

    def _load_image_files(self, directory: str) -> List[str]:
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files.sort()
        return files

    def _load_current_image(self):
        if not self.image_files:
            return
        img_path = os.path.join("radio", self.image_files[self.current_index])
        img = cv2.imread(img_path)
        if img is not None:
            self.original_image = cv2.resize(img, self.config.TARGET_SIZE)
            self.current_image = self.original_image.copy()

    def next_image(self) -> bool:
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self._load_current_image()
            self.scale = 1.0
            return True
        return False

    def previous_image(self) -> bool:
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_image()
            self.scale = 1.0
            return True
        return False

    def zoom_image(self, scale_change: float):
        self.scale = max(self.config.MIN_SCALE,
                        min(self.config.MAX_SCALE, self.scale + scale_change))
        if self.original_image is not None:
            height, width = self.original_image.shape[:2]
            center = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D(center, 0, self.scale)
            self.current_image = cv2.warpAffine(self.original_image, M, (width, height))

    def draw_info(self):
        if self.current_image is not None:
            cv2.putText(self.current_image,
                       f'Image {self.current_index + 1}/{len(self.image_files)}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(self.current_image,
                       f'Zoom: {self.scale:.2f}x',
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def upload_to_firestore(self):
        if self.current_image is not None:
            _, buffer = cv2.imencode('.jpg', self.current_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            try:
                db.collection('zoom-img').add({
                    'image_base': image_base64,
                    'uploaded_at': firestore.SERVER_TIMESTAMP
                })
                print('Zoomed image uploaded to Firestore')
            except Exception as e:
                print(f'Error uploading to Firestore: {e}')

def calculate_distance(p1, p2) -> float:
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

def is_palm_open(landmarks) -> bool:
    finger_tips = [landmarks.landmark[tip] for tip in [8, 12, 16, 20]]
    distances = []
    for i in range(len(finger_tips)-1):
        dist = calculate_distance(finger_tips[i], finger_tips[i+1])
        distances.append(dist)
    return all(d > 0.04 for d in distances)

def mark_position(frame, position: Tuple[int, int], color: Tuple[int, int, int] = (0, 255, 255), radius: int = 10):
    """Mark a position with a circle."""
    cv2.circle(frame, position, radius, color, -1)  # Default: yellow circle

def get_eye_position(hand_landmarks) -> Tuple[int, int]:
    """Get the position of the eye based on the landmarks of the hand."""
    palm_center = hand_landmarks.landmark[9]
    eye_x = int(palm_center.x * controller.original_image.shape[1])
    eye_y = int(palm_center.y * controller.original_image.shape[0])

    return eye_x, eye_y

# Configuration initiale
config = GestureConfig()
detector = HandGestureDetector()
controller = ImageController("radio", config)

gesture_cooldown = 0
previous_pinch_distance = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = detector.process_frame(frame)

    if gesture_cooldown > 0:
        gesture_cooldown -= 1

    if results.multi_hand_landmarks:
        left_hand = right_hand = None

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[hand_idx].classification[0].label
            if handedness == 'Left':
                left_hand = hand_landmarks
            else:
                right_hand = hand_landmarks
            detector.draw_landmarks(frame, hand_landmarks)

        # Circle gesture detection
        if left_hand and detector.detect_circle_gesture(left_hand):
            eye_position = get_eye_position(left_hand)
            mark_position(controller.current_image, eye_position)
            gesture_cooldown = config.COOLDOWN_FRAMES
            cv2.putText(frame, "CIRCLE GESTURE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if right_hand and is_palm_open(right_hand):
            thumb = right_hand.landmark[4]
            index = right_hand.landmark[8]
            pinch_distance = calculate_distance(thumb, index) * 1000

            h, w, _ = frame.shape
            cv2.line(frame,
                    (int(thumb.x * w), int(thumb.y * h)),
                    (int(index.x * w), int(index.y * h)),
                    (0, 0, 255), 2)

            if previous_pinch_distance is not None:
                scale_change = (pinch_distance - previous_pinch_distance) / 200
                controller.zoom_image(scale_change)
                controller.upload_to_firestore()

            cv2.putText(frame, "MODE ZOOM", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            previous_pinch_distance = pinch_distance

    controller.draw_info()
    cv2.imshow('Photo', controller.current_image)
    cv2.imshow('Controle gestuel', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
