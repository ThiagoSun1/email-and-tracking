import cv2
import numpy as np
from ultralytics import YOLO
from adafruit_servokit import ServoKit
import requests
import time
import os
from datetime import datetime
import smtplib
from email.message import EmailMessage
import ssl
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import subprocess
import resampy

# ----------------------------
# EMAIL SETTINGS
# ----------------------------
SENDER_EMAIL = "sender_email@gmail.com"
APP_PASSWORD = "app_password"
RECEIVER_EMAIL = "receiver_email@gmail.com"

# ----------------------------
# PERSON SETTINGS
# ----------------------------
NAME = "patient_name"
AGE = "patient_age"
DISEASE = "patient_disease"

# ----------------------------
# CREATE SAVE FOLDER
# ----------------------------
SAVE_FOLDER = "emergencies"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ----------------------------
# LOAD YOLOv8 POSE MODEL
# ----------------------------
yolo_model = YOLO("yolov8n-pose.pt")

# ----------------------------
# SERVO SETUP
# ----------------------------
kit = ServoKit(channels=16)

PAN_CHANNEL = 0
TILT_CHANNEL = 1

pan_angle = 90
tilt_angle = 90

kit.servo[PAN_CHANNEL].angle = pan_angle
kit.servo[TILT_CHANNEL].angle = tilt_angle

# ----------------------------
# CAMERA SETUP
# ----------------------------
cap = cv2.VideoCapture(0)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ----------------------------
# TRACKING SETTINGS
# ----------------------------
SMOOTHING = 0.2
MAX_STEP = 4
DEADZONE = 30

PAN_DIRECTION = 1
TILT_DIRECTION = -1

# ----------------------------
# VOSK VOICE INPUT SETUP
# ----------------------------
VOSK_MODEL_PATH = "/home/thiago/vosk-model-small-en-us-0.15"
OLLAMA_MODEL = "qwen2:0.5b"

MIC_DEVICE = 0
MIC_RATE = 16000
VOSK_RATE = 16000

audio_queue = queue.Queue()

def mic_callback(indata, frames, time, status):
    if status:
        print(status)
    audio = indata.flatten().astype(np.float32)
    audio_resampled = resampy.resample(audio, MIC_RATE, VOSK_RATE)
    audio_int16 = (audio_resampled * 32767).astype(np.int16)
    audio_queue.put(audio_int16.tobytes())

print("Loading Vosk model...")
vosk_model = Model(VOSK_MODEL_PATH)
rec = KaldiRecognizer(vosk_model, VOSK_RATE)
print("Vosk model loaded.")

# ----------------------------
# ALERT STATE
# ----------------------------
alert_triggered = False

# ----------------------------
# HELPERS
# ----------------------------
def speak(text):
    print("Assistant:", text)
    subprocess.run(["espeak-ng", text])

def ask_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=20
        )
        return response.json()["response"].strip()
    except Exception as e:
        print("Ollama Error:", e)
        return "AI unavailable."

# ----------------------------
# SAVE IMAGE
# ----------------------------
def save_emergency_image(frame, status):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{status.replace(' ', '_')}_{timestamp}.jpg"
    filepath = os.path.join(SAVE_FOLDER, filename)
    cv2.imwrite(filepath, frame)
    print("Saved image:", filepath)

# ----------------------------
# RECORD VIDEO
# ----------------------------
def record_emergency_video(cap, status, duration=10):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{status.replace(' ', '_')}_{timestamp}.mp4"
    filepath = os.path.join(SAVE_FOLDER, filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    print("Saved video:", filepath)
    return filepath

# ----------------------------
# SEND EMAIL
# ----------------------------
def send_email_with_video(video_path, status):
    msg = EmailMessage()
    msg["Subject"] = f"Emergency Alert: {status}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    msg.set_content(f"""
Emergency detected: {status}

Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Video attached.
""")

    with open(video_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(video_path)

    msg.add_attachment(file_data,
                       maintype="video",
                       subtype="mp4",
                       filename=file_name)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)

    print("Email sent successfully.")

# ----------------------------
# VOICE CONVERSATION LOOP
# ----------------------------
def voice_conversation_loop(status, initial_reply):
    """Speak the initial alert reply, then listen and respond until goodbye."""

    speak(initial_reply)

    print(f"Starting voice conversation with {NAME}...")

    with sd.InputStream(
        device=MIC_DEVICE,
        samplerate=MIC_RATE,
        channels=1,
        dtype='float32',
        callback=mic_callback
    ):
        while True:
            # Flush stale audio before listening
            while not audio_queue.empty():
                audio_queue.get()

            print("Listening for response...")
            spoken_text = ""

            while True:
                data = audio_queue.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    spoken_text = result.get("text", "").strip()
                    if spoken_text:
                        break

            print("You said:", spoken_text)

            # Exit conversation on goodbye
            if any(word in spoken_text.lower() for word in ["bye", "goodbye", "stop", "exit"]):
                speak("Okay, take care! Let me know if you need anything.")
                break

            # Context-aware follow-up prompt
            follow_up_prompt = (
                f"You are a helpful assistant talking to {NAME}, who is {AGE} years old "
                f"and has a condition: {DISEASE}. "
                f"A camera previously detected: {status}. "
                f"They just said: '{spoken_text}'. "
                f"Respond helpfully and calmly, addressing {NAME} by name."
            )

            reply = ask_ollama(follow_up_prompt)
            speak(reply)

# ----------------------------
# SERVO TRACKING
# ----------------------------
def track_head(head_x, head_y):
    global pan_angle, tilt_angle

    center_x = FRAME_WIDTH // 2
    center_y = FRAME_HEIGHT // 3

    error_x = head_x - center_x
    error_y = head_y - center_y

    if abs(error_x) < DEADZONE:
        error_x = 0
    if abs(error_y) < DEADZONE:
        error_y = 0

    step_pan = PAN_DIRECTION * (-error_x * 0.04)
    step_tilt = TILT_DIRECTION * (error_y * 0.04)

    step_pan = np.clip(step_pan, -MAX_STEP, MAX_STEP)
    step_tilt = np.clip(step_tilt, -MAX_STEP, MAX_STEP)

    pan_angle += SMOOTHING * step_pan
    tilt_angle += SMOOTHING * step_tilt

    pan_angle = np.clip(pan_angle, 10, 170)
    tilt_angle = np.clip(tilt_angle, 10, 170)

    kit.servo[PAN_CHANNEL].angle = pan_angle
    kit.servo[TILT_CHANNEL].angle = tilt_angle

# ----------------------------
# BEHAVIOR CLASSIFICATION
# ----------------------------
def classify_behavior(person):

    nose = person[0]
    l_shoulder = person[5]
    r_shoulder = person[6]
    l_wrist = person[9]
    r_wrist = person[10]
    l_hip = person[11]
    r_hip = person[12]

    hip_y = (l_hip[1] + r_hip[1]) / 2
    if abs(nose[1] - hip_y) < 50:
        return "FALL DETECTED"

    if (np.linalg.norm(l_wrist - nose) < 60 or
        np.linalg.norm(r_wrist - nose) < 60):
        return "HEAD PAIN"

    stomach_center = np.array([
        (l_hip[0] + r_hip[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2
    ])

    if (np.linalg.norm(l_wrist - stomach_center) < 60 or
        np.linalg.norm(r_wrist - stomach_center) < 60):
        return "STOMACH PAIN"

    back_center = np.array([
        (l_shoulder[0] + r_shoulder[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2
    ])

    if (np.linalg.norm(l_wrist - back_center) < 60 and
        np.linalg.norm(r_wrist - back_center) < 60):
        return "BACK PAIN"

    return "OK"

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, verbose=False)
    status = "No Person"

    if len(results) > 0 and results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()

        if len(keypoints) > 0:
            person = keypoints[0]

            for x, y in person.astype(int):
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            head_x = int(person[0][0])
            head_y = int(person[0][1])
            track_head(head_x, head_y)

            status = classify_behavior(person)

    # ----------------------------
    # EMERGENCY TRIGGER
    # ----------------------------
    if status not in ["OK", "No Person"] and not alert_triggered:

        print("ALERT:", status)

        # Save image + record video + send email
        save_emergency_image(frame, status)
        video_path = record_emergency_video(cap, status, duration=10)
        send_email_with_video(video_path, status)

        # Generate initial AI response
        prompt = (
            f"Use {NAME} throughout the conversation so they know it's them. "
            f"A camera detected a person named {NAME} has {status}. "
            f"Calmly ask if {NAME} needs help. "
            f"For information, {NAME} is {AGE} years old and has {DISEASE}. "
            f"Just talk to them based on the information I just gave you."
        )

        initial_reply = ask_ollama(prompt)
        print("AI:", initial_reply)

        # Start voice conversation (speak + listen + reply loop)
        voice_conversation_loop(status, initial_reply)

        alert_triggered = True

    if status in ["OK", "No Person"]:
        alert_triggered = False

    cv2.putText(frame, status, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 3)

    cv2.imshow("Tracking + Emergency System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# CLEANUP
# ----------------------------
cap.release()
cv2.destroyAllWindows()
