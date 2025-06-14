import os
import argparse
import torch
import torch.nn as nn
from datareader import YoutubeCommentDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import TextCNN
import numpy as np
import random
import matplotlib.pyplot as plt

# ==== Hyperparameters ====
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-4
EMBED_DIM = 300
NUM_CLASSES = 2

# ==== Model Factory Function ====
def get_model(model_name, vocab_size, embed_dim, num_classes):
    if model_name.lower() == "textcnn":
        return TextCNN(vocab_size=vocab_size, embed_dim=embed_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

# ==== Optimizer Factory Function ====
def get_optimizer(optimizer_name, model_params, lr):
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adam":
        return torch.optim.Adam(model_params, lr=lr)
    elif optimizer_name == "adadelta":
        return torch.optim.Adadelta(model_params, lr=lr)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model_params, lr=lr)
    elif optimizer_name == "adagrad":
        return torch.optim.Adagrad(model_params, lr=lr)
    elif optimizer_name == "adamax":
        return torch.optim.Adamax(model_params, lr=lr)
    elif optimizer_name == "asgd":
        return torch.optim.ASGD(model_params, lr=lr)
    elif optimizer_name == "nadam":
        return torch.optim.NAdam(model_params, lr=lr)
    elif optimizer_name == "radam":
        return torch.optim.RAdam(model_params, lr=lr)
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model_params, lr=lr)
    elif optimizer_name == "rprop":
        return torch.optim.Rprop(model_params, lr=lr)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")

# ==== Training Function ====
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(data_loader), correct / total

# ==== Evaluation Function ====
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(data_loader), correct / total

# ==== Main Training ====
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, default="textcnn", help="Model name")
    parser.add_argument("--optimizer_name", type=str, default="adam", help="Optimizer name")
    parser.add_argument("--augment_prob", type=float, default=1.0, help="Probability of applying augmentation")
    args = parser.parse_args()

    seed = 2003
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs('output', exist_ok=True)

    train_dataset = YoutubeCommentDataset(
        file_path="dataset/dataset_judol.xlsx",
        # tokenizer_name="D:/TA/code/indobert-base-p1",
        tokenizer_name="indobenchmark/indobert-base-p1",
        folds_file="youtube_datareader-simple-folds.json",
        random_state=2003,
        split="train",
        fold=0,
        augmentasi_file="augmentasi.json",
        augment_prob=args.augment_prob
    )
    val_dataset = YoutubeCommentDataset(
        file_path="dataset/dataset_judol.xlsx",
        # tokenizer_name="D:/TA/code/indobert-base-p1",
        tokenizer_name="indobenchmark/indobert-base-p1",
        folds_file="youtube_datareader-simple-folds.json",
        random_state=2003,
        split="val",
        fold=0,
        augmentasi_file="augmentasi.json",
        augment_prob=0.0
    )

    vocab_size = train_dataset.tokenizer.vocab_size

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("textcnn", vocab_size, EMBED_DIM, NUM_CLASSES).to(device)


    optimizer = get_optimizer(args.optimizer_name, model.parameters(), LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'output/best_model.pth')

    epochs = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
