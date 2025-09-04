import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
import shutil
from facenet_pytorch import MTCNN, InceptionResnetV1

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
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
else:
    known_face_encodings = []
    known_face_names = []

print(f"[INFO] Total known faces loaded: {len(set(known_face_names))}")

# ================================
# Save embeddings helper
# ================================
def save_embeddings():
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
    print("[INFO] Embeddings updated.")

# ================================
# Delete face function
# ================================
def delete_face(name):
    global known_face_encodings, known_face_names

    if name not in known_face_names:
        print(f"[WARN] No face found with name '{name}'.")
        return

    # Remove embeddings for this name
    indices_to_keep = [i for i, n in enumerate(known_face_names) if n != name]
    known_face_encodings = [known_face_encodings[i] for i in indices_to_keep]
    known_face_names = [known_face_names[i] for i in indices_to_keep]

    # Save updated embeddings
    save_embeddings()

    # Remove folder
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
        print(f"[INFO] Deleted folder {person_dir}")

    print(f"[INFO] Successfully deleted face '{name}'.")

# ================================
# Start Webcam
# ================================
cap = cv2.VideoCapture(0)
detected_faces = set()
saving_mode = False
save_count = 0
save_name = None

while True:
    ret, frame = cap.read()
    if not ret:
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
                    distances = [np.linalg.norm(embedding - known) for known in known_face_encodings]
                    min_distance = min(distances)
                    best_match = np.argmin(distances)

                    if min_distance < 0.75:  # threshold
                        name = known_face_names[best_match]

                x1, y1, x2, y2 = coords[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if name != "Unknown":
                    current_faces.add(name)

                # ================================
                # Save new faces (up to 30)
                # ================================
                if saving_mode and save_count < 30:
                    person_dir = os.path.join(KNOWN_FACES_DIR, save_name)
                    os.makedirs(person_dir, exist_ok=True)
                    img_path = os.path.join(person_dir, f"{save_name}_{save_count}.jpg")

                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size != 0:
                        cv2.imwrite(img_path, face_img)

                        known_face_encodings.append(embedding)
                        known_face_names.append(save_name)

                        save_count += 1
                        print(f"[INFO] Saved {img_path}")

                    if save_count == 30:
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

    # ================================
    # Key controls
    # ================================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not saving_mode:
        save_name = input("Enter name/USN for this face: ")
        saving_mode = True
        save_count = 0
        print(f"[INFO] Saving up to 30 images for {save_name}...")

    if key == ord('d'):  # ✅ delete mode
        name_to_delete = input("Enter the name to delete: ")
        delete_face(name_to_delete)

    if key == ord('q'):
        break

    cv2.imshow("Face Recognition", frame)

cap.release()
cv2.destroyAllWindows()
