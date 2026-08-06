"""
File: hand_detector.py

Purpose:
Uses Google's MediaPipe library to detect hands in a camera frame.

Input:
    A single frame captured by OpenCV.

Output:
    MediaPipe detection results containing hand landmarks.
"""

import cv2
import mediapipe as mp


class HandDetector:

    def __init__(self):
        # Load Google's MediaPipe hand detection tools.
        self.mp_hands = mp.solutions.hands

        # Create and configure the hand detector.
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Utility used later to draw the hand landmarks.
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        # Convert OpenCV's BGR image to RGB because
        # MediaPipe expects RGB images.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Analyze the frame and return MediaPipe's detection results.
        results = self.hands.process(rgb_frame)

        #print wrist landmarks
        if results.multi_hand_landmarks:
            hand=results.multi_hand_landmarks[0]

            wrist=hand.landmark[0]

            #print(
            #    f"Wrist -> "
            #   f"x: {wrist.x:.3f}, "
            #    f"y: {wrist.y:.3f}, "
            #    f"z: {wrist.z:.3f}"
            #)

        return results

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            #loop through detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                #draw the 21 landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )