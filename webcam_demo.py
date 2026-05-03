import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from ema_pytorch import EMA
from models import get_model
import mediapipe as mp
from huggingface_hub import hf_hub_download

checkpoint_repo = "twuan/EmoNeXt-FER-Optimized"
checkpoint_file = "EmoNeXt_tiny.pt"
checkpoint_path = hf_hub_download(repo_id=checkpoint_repo, filename=checkpoint_file)
MODEL_SIZE = "tiny"
IN_22K = True

classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Val-like transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize(236),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])

net = get_model(len(classes), MODEL_SIZE, in_22k=IN_22K).to(device)
ema = EMA(net).to(device)

data = torch.load(checkpoint_path, map_location=device)
net.load_state_dict(data["model"])
ema.load_state_dict(data["ema"])
ema.eval()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

mp_face = mp.solutions.face_detection
face_det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_det.process(rgb)
    
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
    
            # Padding nhe
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
    
            face_roi = rgb[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
            
            inp = transform(face_roi).unsqueeze(0).to(device)
    
            with torch.no_grad():
                _, logits = ema(inp)
    
            probs = F.softmax(logits, dim=1).squeeze(0)
            top_idx = int(torch.argmax(probs))
            top_emotion = classes[top_idx]
            top_pct = float(probs[top_idx] * 100.0)
    
            label = f"{top_emotion}: {top_pct:.1f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    else:
        cv2.putText(frame, "No face", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    
    cv2.imshow("EmoNeXt Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()