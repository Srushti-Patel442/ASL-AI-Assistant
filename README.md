# ASL AI Assistant

A real-time desktop application that recognizes American Sign Language (ASL), translates recognized signs into natural English using a locally hosted large language model, and generates speech through offline text-to-speech.

---

## Overview

ASL AI Assistant is an end-to-end computer vision application designed to bridge communication between ASL users and non-signers. Using a webcam, the application detects hand landmarks with MediaPipe, classifies both static letters and dynamic gestures using custom TensorFlow models, translates recognized signs into fluent English with Llama 3 running locally through Ollama, and generates speech using an offline text-to-speech engine.

The entire inference pipeline runs locally without requiring cloud services or internet connectivity.

---

## Features

- Real-time ASL recognition from a live webcam feed
- Static ASL letter classification using TensorFlow/Keras
- Dynamic gesture recognition using temporal sequence modeling
- Natural language translation powered by a local Llama 3 model
- Offline text-to-speech generation
- Live confidence scoring and prediction visualization
- Interactive phrase builder for multi-sign translation

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Computer Vision | OpenCV, MediaPipe |
| Machine Learning | TensorFlow, Keras, NumPy |
| Natural Language Processing | Ollama, Llama 3 |
| Speech Synthesis | pyttsx3 |
| Version Control | Git, GitHub |

---

## Dataset

The recognition models are trained using custom datasets collected with MediaPipe hand landmarks.

### Static Recognition

- 26 ASL alphabet classes
- Target dataset: **300 samples per letter**

### Dynamic Recognition

- 8 gesture classes
- Target dataset: **150–200 gesture sequences per class**

Current supported gestures include:

- Hello
- Yes
- No
- Thank You
- Please
- Sorry
- I Love You

Additional gestures are currently being developed.

---

## Controls

| Key | Action |
|------|--------|
| **Space** | Save the current recognized sign |
| **Enter** | Translate the current phrase and generate speech |
| **C** | Clear the current phrase |
| **Q** | Exit the application |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Srushti-Patel442/ASL-AI-Assistant.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m backend.app.vision.camera
```

---

## Roadmap

- Expand the supported ASL vocabulary
- Improve recognition accuracy with larger training datasets
- Add conversational context for more natural translations
- Enhance the desktop interface and user experience
- Explore cloud-based deployment

---

## Author

**Srushti Patel**
Toronto Metropolitan University

GitHub: https://github.com/Srushti-Patel442
