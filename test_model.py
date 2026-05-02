import sys
import torch
import numpy as np
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from models import get_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix 

CHECKPOINT_PATH = "rs_trained_opt\\checkpoint\\EmoNeXt_tiny_2026-04-30_08-43-15.pt" 
DATASET_PATH = "FER2013"
MODEL_SIZE = "tiny"
IN_22K = True
BATCH_SIZE = 32  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device.type)

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(236),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack(
        [transforms.ToTensor()(crop) for crop in crops]
    )),
    transforms.Lambda(lambda crops: torch.stack(
        [crop.repeat(3, 1, 1) for crop in crops]
    )),
])

test_dataset = datasets.ImageFolder(DATASET_PATH + "/test", test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
classes = test_dataset.classes
print(f"Testing on {len(test_dataset)} images, {len(classes)} classes")

# Load model
net = get_model(len(classes), MODEL_SIZE, in_22k=IN_22K)
net = net.to(device)

ema = EMA(net).to(device)

data = torch.load(CHECKPOINT_PATH, map_location=device)
net.load_state_dict(data["model"])
ema.load_state_dict(data["ema"])
print(f"Loaded checkpoint, best val acc: {data['best_acc']:.4f}%")

# Test
ema.eval()
predicted_labels = []
true_labels = []

pbar = tqdm(unit="batch", file=sys.stdout, total=len(test_loader))
for inputs, labels in test_loader:
    bs, ncrops, c, h, w = inputs.shape
    inputs = inputs.view(-1, c, h, w).to(device)
    labels = labels.to(device)

    with torch.no_grad():
        with torch.autocast(device.type, enabled=True):
            _, logits = ema(inputs)

    outputs_avg = logits.view(bs, ncrops, -1).mean(1)
    predictions = torch.argmax(outputs_avg, dim=1)

    predicted_labels.extend(predictions.tolist())
    true_labels.extend(labels.tolist())
    pbar.update(1)

pbar.close()

accuracy = (
    torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels))
    .float().mean().item()
)
print(f"\nTest Accuracy: {accuracy * 100:.4f}%")

# Per-class accuracy
print("\nPer-class accuracy:")
for i, cls in enumerate(classes):
    idx = [j for j, l in enumerate(true_labels) if l == i]
    correct = sum(predicted_labels[j] == true_labels[j] for j in idx)
    print(f"  {cls}: {correct}/{len(idx)} = {correct/len(idx)*100:.1f}%")

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(len(classes))))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Test")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png", dpi=200)
plt.close()
print("Saved: confusion_matrix_test.png")