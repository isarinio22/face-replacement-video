import cv2
import numpy as np
from mtcnn import MTCNN
import subprocess
import os

# === Inputs ===
video_path = "AXE_READY.mp4"
output_path_no_audio = "output_no_audio.mp4"
final_output_path = "output_with_audio.mp4"

face1_img = cv2.imread("ISAR.png", cv2.IMREAD_UNCHANGED)
face2_img = cv2.imread("CYBERARK.png", cv2.IMREAD_UNCHANGED)
axe_replace_img = cv2.imread("HUGGING_FACE.png", cv2.IMREAD_UNCHANGED)
axe_template = cv2.imread("AXE.png")

# === MTCNN face detector ===
detector = MTCNN()

# === Video IO ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path_no_audio, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

def overlay(frame, img, x, y, w, h):
    if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
        return frame
    img = cv2.resize(img, (w, h))
    if img.shape[2] == 4:
        alpha = img[:,:,3] / 255.0
        for c in range(3):
            frame[y:y+h, x:x+w, c] = alpha*img[:,:,c] + (1-alpha)*frame[y:y+h, x:x+w, c]
    else:
        frame[y:y+h, x:x+w] = img
    return frame

frame_count = 0
skip_frames = 2  # detect faces every 2 frames
last_faces = []  # store last detected bounding boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # === Face detection every skip_frames ===
    if frame_count % skip_frames == 1:
        faces = detector.detect_faces(rgb_frame)
        last_faces = faces  # store detected faces
    else:
        faces = last_faces  # reuse previous detection

    # Overlay faces
    for i, face in enumerate(faces[:2]):  # overlay only first 2 faces
        x, y, w, h = face['box']
        x = max(0, x)
        y = max(0, y)
        w = min(w, W - x)
        h = min(h, H - y)
        if i == 0:
            frame = overlay(frame, face1_img, x, y, w, h)
        else:
            frame = overlay(frame, face2_img, x, y, w, h)

    # === Axe detection (template matching) ===
    res = cv2.matchTemplate(frame, axe_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val > 0.6:
        x, y = max_loc
        th, tw, _ = axe_template.shape
        frame = overlay(frame, axe_replace_img, x, y, tw, th)

    out.write(frame)

    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print("✅ Video frames processed. Adding audio...")

# === Merge original audio using ffmpeg ===
if not os.path.exists(output_path_no_audio):
    raise FileNotFoundError("Processed video not found to merge audio.")

ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-i", output_path_no_audio,
    "-i", video_path,
    "-c", "copy",
    "-map", "0:v:0",
    "-map", "1:a:0",
    final_output_path
]

subprocess.run(ffmpeg_cmd, check=True)
print(f"Final video saved with original audio at: {final_output_path}")
