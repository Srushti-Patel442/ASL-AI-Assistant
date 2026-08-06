"""
File: camera.py

Purpose:
Connects to the computer's webcam and displays a live video feed.

First step of the computer vision pipeline.
Later, each video frame will be passed to MediaPipe for hand detection
and then to our AI model for gesture recognition.
"""

import cv2
import time
import threading
from app.vision.hand_detector import HandDetector
from app.ai.predict import ASLPredictor
from app.dynamic.predict_gesture import GesturePredictor
from app.llm.llm_client import LLMTranslator
from app.speech.tts import Speaker

def start_camera():
    camera = cv2.VideoCapture(0)# 0 is the built in webcam
    detector=HandDetector()#create a hand detector for entire program (load mediapipe)
    predictor=ASLPredictor()#load trained AI
    gesture_predictor=GesturePredictor()
    translator=LLMTranslator()
    speaker=Speaker()

    gesture_buffer=[]

    recognized_signs=[]
    translated_signs=""
    
    gesture_prediction=""
    gesture_confidence=0.0
    last_gesture_time=0
    stable_prediction=""
    stable_confidence=0.0

    translating=False
    def translate_and_speak(signs):
         nonlocal translated_signs
         nonlocal translating

         try:
            translated_signs=translator.translate(signs)
            speaker.speak(translated_signs)
         except Exception as e:
              print("Translation thread error:", e)
         finally:
              translating=False
    
         translating=False

    while True:
        # Check if the frame is captured from the webcam
        # Read one frame from the webcam
        success, frame = camera.read()
        if not success:
                    print("Failed to read from webcam.")
                    break
        
        results=detector.detect(frame)#MediaPipe anaylzes current frame

        if results.multi_hand_landmarks:
            hand=results.multi_hand_landmarks[0]
            sample = []

            for landmark in hand.landmark:
                sample.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            gesture_buffer.append(sample)

            if len(gesture_buffer)>60:
                gesture_buffer.pop(0)

            if len(gesture_buffer)==60:
                 gesture_prediction, gesture_confidence=gesture_predictor.predict(
                      gesture_buffer.copy()
                 )
                 last_gesture_time=time.time()
                 gesture_buffer.clear()

            prediction, confidence=predictor.predict(sample)

            if (
                 gesture_confidence>0.90
                and time.time()-last_gesture_time<1
            ):
                 prediction=gesture_prediction
                 confidence=gesture_confidence

            stable_prediction=prediction
            stable_confidence=confidence
            
            if confidence<0.70:
                stable_prediction="Unknown"
                stable_confidence=confidence

            cv2.putText(
                frame,
                "ASL AI Assistant",
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 235, 200),
                2
            )
            
            cv2.putText(
                frame,
                f"Prediction: {stable_prediction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 235, 200),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {stable_confidence * 100:.1f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 235, 200),
                2
            )

            cv2.putText(
                frame,
                "Recognized:",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 235, 200),
                2
            )

            y=150

            for sign in recognized_signs[-5:]:
                cv2.putText(
                    frame,
                    sign,
                    (40,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 235, 200),
                    2
                )
                y+=30

            cv2.putText(
                frame,
                "English:",
                (20, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 235, 200),
                2
            )

            cv2.putText(
                frame,
                translated_signs,
                (20, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 235, 200),
                2
            )

            cv2.putText(
                frame,
                "SPACE=Save   ENTER=Translate   C=Clear   Q=Quit",
                (20, 430),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 235, 200),
                2
            )

        else:
            gesture_buffer.clear()
            gesture_confidence=0.0
            gesture_prediction=""

        detector.draw_landmarks(frame, results)#draw landmark

        # Display the current frame
        cv2.imshow("ASL AI Assistant", frame)

        # Press 'q' to quit
        # wait 1 millisecond for the key to press
        key=cv2.waitKey(1) & 0xFF

        if key==ord(" "):
             if(
                  stable_prediction!="Unknown"
                  and stable_confidence>0.70 
             ):
                  if(
                       len(recognized_signs)==0
                       or recognized_signs[-1]!=stable_prediction
                  ):
                    recognized_signs.append(stable_prediction)
                    print(recognized_signs)

                    gesture_prediction="" 
                    gesture_confidence=0.0
                    gesture_buffer.clear()

        elif key == 13: #Enter
            if recognized_signs and not translating:
                 translating=True
                 translated_signs="Translating..."
                 threading.Thread(
                      target=translate_and_speak,
                      args=(recognized_signs.copy(),),
                      daemon=True
                 ).start()

        elif key==ord("c"):
             recognized_signs.clear()
             translated_signs=""

             stable_prediction=""
             stable_confidence=0.0

             gesture_prediction="" 
             gesture_confidence=0.0
             gesture_buffer.clear()

        elif key==ord("q"):
             break

    camera.release()#release webcam so other programs can use it
    cv2.destroyAllWindows()#close all OpenCV windows

#run the program only if this file is executed
if __name__ == "__main__":
    start_camera()