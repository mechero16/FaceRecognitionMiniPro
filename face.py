import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
import shutil
from facenet_pytorch import MTCNN, InceptionResnetV1
import json
import platform

# ================================
# Setup
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

KNOWN_FACES_DIR = "known_faces"
EMBEDDINGS_FILE = "embeddings.pkl"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Load FaceNet models
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# ================================
# Load Known Faces Embeddings
# ================================
if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    # Check if the loaded data is a dictionary with an 'encodings' key
    if isinstance(data, dict) and 'encodings' in data:
        known_face_encodings = data["encodings"]
    else:
        # If it's an old format (e.g., a simple list), treat it as empty
        # and let the user re-register faces.
        known_face_encodings = {}
        print("[WARN] Old or corrupted embeddings file detected. Starting with no known faces.")
        # Optionally, you could delete the old file to force a clean start
        # os.remove(EMBEDDINGS_FILE)
else:
    known_face_encodings = {}

print(f"[INFO] Total known faces loaded: {len(known_face_encodings)}")

# ================================
# Save embeddings helper
# ================================
def save_embeddings():
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_face_encodings}, f)
    print("[INFO] Embeddings updated.")

# ================================
# Delete face function
# ================================
def delete_face(name):
    global known_face_encodings

    if name not in known_face_encodings:
        print(f"[WARN] No face found with name '{name}'.")
        return

    del known_face_encodings[name]

    # Save updated embeddings
    save_embeddings()

    # Remove folder
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
        print(f"[INFO] Deleted folder {person_dir}")

    print(f"[INFO] Successfully deleted face '{name}'.")

# ================================
# Cosine similarity helper
# ================================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# ================================
# Camera Selection (Cross-Platform)
# ================================
print("[INFO] Searching for available cameras...")
available_cameras = []
os_name = platform.system()

for i in range(10):  # Check a range of common camera indices
    cap = None
    try:
        # Try without a specific backend first
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            # If that fails, try a specific backend based on the OS
            if os_name == "Windows":
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            elif os_name == "Linux":
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            # Default to no backend if not Windows/Linux
            else:
                cap = cv2.VideoCapture(i)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    except Exception as e:
        print(f"Error checking camera {i}: {e}")
        if cap:
            cap.release()
        continue

if not available_cameras:
    print("[ERROR] No cameras found. Please check your camera connections.")
    exit()

print(f"[INFO] Found the following cameras: {available_cameras}")

camera_index = -1
while camera_index not in available_cameras:
    try:
        camera_index = int(input("Please enter the index of the camera you want to use: "))
        if camera_index not in available_cameras:
            print("[WARN] Invalid camera index. Please choose from the available cameras.")
    except ValueError:
        print("[WARN] Invalid input. Please enter a number.")

# ================================
# Start Webcam
# ================================
if os_name == "Windows":
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
elif os_name == "Linux":
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
else:
    cap = cv2.VideoCapture(camera_index)

detected_faces = set()
saving_mode = False
save_count = 0
save_name = None
temp_embeddings = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from camera. Exiting.")
        break

    # Detect faces
    boxes, probs = mtcnn.detect(frame)
    current_faces = set()

    if boxes is not None:
        faces = []
        coords = []
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_tensor = torch.tensor(face_rgb).permute(2, 0, 1).float() / 255
            face_tensor = F.interpolate(face_tensor.unsqueeze(0), size=(160, 160),
                                         mode="bilinear", align_corners=False)
            faces.append(face_tensor)
            coords.append((x1, y1, x2, y2))

        if len(faces) > 0:
            faces = torch.cat(faces).to(device)
            with torch.no_grad():
                embeddings = resnet(faces).cpu().numpy()

            for i, embedding in enumerate(embeddings):
                name = "Unknown"

                if len(known_face_encodings) > 0:
                    similarities = {
                        person: cosine_similarity(embedding, known_emb)
                        for person, known_emb in known_face_encodings.items()
                    }
                    best_match = max(similarities, key=similarities.get)
                    best_score = similarities[best_match]

                    if best_score > 0.65:  # cosine similarity threshold
                        name = best_match

                x1, y1, x2, y2 = coords[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if name != "Unknown":
                    current_faces.add(name)

                # ================================
                # Save new faces (collect embeddings first)
                # ================================
                if saving_mode and save_count < 30:
                    person_dir = os.path.join(KNOWN_FACES_DIR, save_name)
                    os.makedirs(person_dir, exist_ok=True)
                    img_path = os.path.join(person_dir, f"{save_name}_{save_count}.jpg")

                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size != 0:
                        cv2.imwrite(img_path, face_img)

                        temp_embeddings.append(embedding)
                        save_count += 1
                        print(f"[INFO] Saved {img_path}")

                    if save_count == 30:
                        # Store average embedding for this person
                        avg_embedding = np.mean(temp_embeddings, axis=0)
                        known_face_encodings[save_name] = avg_embedding
                        temp_embeddings = []
                        saving_mode = False
                        save_embeddings()
                        print(f"[INFO] Finished saving 30 images & embeddings for {save_name}")

    # ================================
    # Print only if visible set changes
    # ================================
    if current_faces != detected_faces:
        detected_faces = current_faces.copy()
        if detected_faces:
            print("Visible faces:", detected_faces)
            with open("data.json", "w") as json_file:
                json.dump(list(detected_faces), json_file, indent=4)

    # ================================
    # Key controls
    # ================================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not saving_mode:
        save_name = input("Enter name/USN for this face: ")
        saving_mode = True
        save_count = 0
        temp_embeddings = []
        print(f"[INFO] Saving up to 30 images for {save_name}...")

    if key == ord('d'):  # ✅ delete mode
        name_to_delete = input("Enter the name to delete: ")
        delete_face(name_to_delete)

    if key == ord('q'):
        break

    cv2.imshow("Face Recognition", frame)

cap.release()
cv2.destroyAllWindows()