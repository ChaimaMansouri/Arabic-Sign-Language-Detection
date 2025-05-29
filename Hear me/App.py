from flask import Flask, render_template, Response, jsonify
from arabic_reshaper import reshape
import os
import numpy as np
import cv2
import mediapipe as mp
import joblib 
from gtts import gTTS
import time
from bidi.algorithm import get_display
import base64
import json

app = Flask(__name__)

# Chargement du modèle
model = joblib.load("random_forest_model.pkl")

# Mapping des labels arabes
label_to_arabic = {
    "Ain": "ع", "Al": "ٱ", "Alef": "ا", "Beh": "ب", "Dad": "ض", "Dal": "د",
    "Feh": "ف", "Ghain": "غ", "Hah": "ح", "Heh": "ھ", "Jeem": "ج", "Kaf": "ك",
    "Khah": "خ", "Laa": "لا", "Lam": "ل", "Meem": "م", "Noon": "ن", "Qaf": "ق",
    "Reh": "ر", "Sad": "ص", "Seen": "س", "Sheen": "ش", "Tah": "ط", "Teh": "ت",
    "Teh_Marbuta": "ة", "Thal": "ذ", "Theh": "ث", "Waw": "و", "Yeh": "ي",
    "Zah": "ظ", "Zain": "ز"
}

# Initialisation MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Variables globales
current_letter = ""
confirmed_word = ""
last_stable_label = None
label_hold_start = None
hold_duration_required = 2

cap = cv2.VideoCapture(0)

def generate_frames():
    global current_letter, confirmed_word, last_stable_label, label_hold_start

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        progress = 0
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                )

                landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
                if len(landmarks) == 63:
                    input_data = np.array(landmarks).reshape(1, -1)
                    predicted_label = model.predict(input_data)[0].strip().capitalize()
                    current_letter = label_to_arabic.get(predicted_label, "?")

                    now = time.time()
                    if predicted_label != last_stable_label:
                        last_stable_label = predicted_label
                        label_hold_start = now
                    else:
                        if label_hold_start:
                            progress = min((now - label_hold_start) / hold_duration_required, 1)
                            if (now - label_hold_start) >= hold_duration_required:
                                confirmed_word += current_letter
                                last_stable_label = None
                                label_hold_start = None

        # Encodage de l'image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Préparation des données
        data = {
            "frame": frame_base64,
            "letter": current_letter,
            "progress": progress,
            "word": confirmed_word
        }

        yield f"data: {json.dumps(data)}\n\n"

@app.route('/')
def index1():
    return render_template('index.html')

@app.route('/detection')
def index():
    return render_template('camera.html')

@app.route('/detection/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive'})

@app.route('/detection/get_word')
def get_word():
    return jsonify({"word": confirmed_word})

@app.route('/detection/reset_word')
def reset_word():
    global confirmed_word, current_letter, last_stable_label, label_hold_start
    confirmed_word = ""
    current_letter = ""
    last_stable_label = None
    label_hold_start = None
    return jsonify({"status": "success"})

from playsound import playsound

@app.route('/detection/speak_word')
def speak_word():
    global confirmed_word
    if confirmed_word.strip():
        tts = gTTS(text=confirmed_word, lang='ar')
        temp_filename = "temp_arabic.mp3"
        tts.save(temp_filename)

        try:
            playsound(temp_filename)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})

        os.remove(temp_filename)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Empty word"})

@app.route('/detection/undo_letter')
def undo_letter():
    global confirmed_word
    if len(confirmed_word) > 0:
        confirmed_word = confirmed_word[:-1]
        return jsonify({"status": "success", "word": confirmed_word})
    return jsonify({"status": "error", "message": "Nothing to undo"})

@app.route('/check_camera')
def check_camera():
    if cap.isOpened():
        return jsonify({"status": "success", "message": "Camera is working"})
    return jsonify({"status": "error", "message": "Could not open camera"})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)