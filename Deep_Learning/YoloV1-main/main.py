"""
Main file for training Yolo model on Pascal VOC dataset
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VOCDataset
from model import YoloNet
from yolo_loss import Yolo_loss

# -------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 2e-5
BATCH_SIZE = 64
EPOCHS = 1000
NUM_WORKERS = 4

IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
CSV_TRAIN = "data/100examples.csv"
CSV_TEST = "data/test.csv"
S = 7
B = 2
C = 20
# -------------------------------------------------------------------- #
seed = 42
torch.manual_seed(seed)

def main():
    # Network
    model = YoloNet(in_c=3, S=7, C=20, B=2).to(device)
    # Adam
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    # Loss
    criterion = Yolo_loss()
    # Dataloader
    train_dataset = VOCDataset(csv_file=CSV_TRAIN, img_dir=IMG_DIR, label_dir=LABEL_DIR)
    test_dataset = VOCDataset(csv_file=CSV_TEST, img_dir=IMG_DIR, label_dir=LABEL_DIR)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )

    # Training Loop
    for epoch in range(EPOCHS):
        mean_loss = []
        for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)
            preds = model(x)
            # predictions shape = Batch_size, S, S, C+B*5
            preds = preds.reshape(-1, S, S, C + (B * 5))

            loss = criterion(preds, y)
            mean_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Mean Loss: {sum(mean_loss)/len(mean_loss)}")

# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #

if __name__ == "__main__":
    main()