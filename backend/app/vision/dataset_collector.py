"""
File: dataset_collector.py

Purpose:
Collects hand landmark data from MediaPipe and saves it
to a CSV file for AI training.
"""
import cv2
import csv
import os
import time
import numpy as np

from app.vision.hand_detector import HandDetector


def collect_dataset():

    camera=cv2.VideoCapture(0)
    detector=HandDetector()

    print("----------ASL Dataset Collector----------")
    while True:
        print("\nSelect Mode")
        print("1. Static Letter")
        print("2. Dynamic Gesture")
        mode=input("Choice: ").strip()
        if mode in ["1", "2"]:
            break
        print("Invalid choice.\n")

    if mode=="1":
        while True:
            current_letter=input(
                "Enter a letter (A-Z) that you want to collect: "
            ).strip().upper()
            if len(current_letter)==1 and current_letter.isalpha():
                break
            print("Invalid input. Please enter a single letter from A-Z.\n")
    else:
        while True:
            current_letter=input( 
                "Enter gesture name: " 
            ).strip().upper()
            if current_letter:
                break 
            print("Gesture name cannot be empty.\n")
    
    if mode=="1":
        folder_path=f"../dataset/static/{current_letter}"
        csv_path=f"{folder_path}/{current_letter}.csv"
    else:
        folder_path=f"../dataset/dynamic/{current_letter}"
    os.makedirs(folder_path, exist_ok=True)

    #count how many samples already exist
    if mode == "1":
        if os.path.exists(csv_path):
            with open(csv_path, "r") as file:
                sample_count = sum(1 for _ in file)
        else:
            sample_count=0
    else:
        sample_count=0
        for file in os.listdir(folder_path):
            if file.endswith(".npy"):
                sample_count+=1

    show_saved_message=False
    message_frames=0

    state="idle"
    countdown_start=0
    gesture=[]

    while True:

        success, frame=camera.read()

        if not success:
            print("Failed to read from webcam.")
            break

        results=detector.detect(frame)

        if results.multi_hand_landmarks:
            hand=results.multi_hand_landmarks[0]
            #for landmark in hand.landmark:
            #   print(landmark.x, landmark.y, landmark.z)

        detector.draw_landmarks(frame, results)

        if mode=="1":
            display_text=f"Letter: {current_letter}"
        else:
            display_text=f"Gesture: {current_letter}"

        cv2.putText(
            frame,
            display_text,
            (20,35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (200, 200, 200),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}",
            (20,70),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (200, 200, 200),
            2
        )

        if mode=="1":
            controls="S = Save   Q = Quit"
        else:
            controls="R = Record   Q = Quit"

        cv2.putText(
            frame,
            controls,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            2   
        )

        if show_saved_message:
            cv2.putText(
                frame,
                "Sample saved!",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 200, 200),
                2
            )
            message_frames-=1
            if message_frames<=0:
                show_saved_message=False

        if state=="countdown":
            elapsed=time.time()-countdown_start
            countdown_number=3-int(elapsed)
            if countdown_number>0:
                cv2.putText(
                    frame,
                    str(countdown_number),
                    (250,250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (255,255,255),
                    6
                )
            else:
                state="recording"
                gesture=[]
                print("Recording...")

        if state=="recording":
            if results.multi_hand_landmarks:
                hand=results.multi_hand_landmarks[0]
                sample=[]
                for landmark in hand.landmark:
                    sample.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])
                gesture.append(sample)
            if len(gesture)>=60:
                print("Finished Recording!")
                print(f"Frames captured: {len(gesture)}")
                gesture_array=np.array(gesture,dtype=np.float32)
                save_path=os.path.join(
                    folder_path,
                    f"sample{sample_count+1:03d}.npy"#:03d means to use 3 digits
                )
                np.save(save_path, gesture_array)
                print(f"Saved to {save_path}")
                sample_count+=1
                show_saved_message=True
                message_frames=30
                state="idle"

        cv2.imshow("Dataset collector", frame)
        key=cv2.waitKey(1)&0xFF

        if mode=="1":
            if key==ord("s"):
                if results.multi_hand_landmarks:
                    hand=results.multi_hand_landmarks[0]
                    sample=[]
                    #collect all 63 values
                    for landmark in hand.landmark:
                        sample.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

                    #save a row in A.csv
                    with open(csv_path, "a", newline="") as file:
                        writer=csv.writer(file)
                        writer.writerow(sample)

                    sample_count+=1
                    
                    show_saved_message=True
                    message_frames=30

                    print (f"Saved sample #{sample_count} for letter {current_letter}")
        else:
            if key==ord("r") and state=="idle":
                state="countdown"
                countdown_start=time.time()

        if key==ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_dataset()

