# multistream_finbert_training_routine.py
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm

def train_model(model, dataset, val_dataset, loss_fn, epochs=10, lr=3e-4, batch_size=16, device='cuda'):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

        evaluate_model(model, val_loader, device)

def evaluate_model(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            pass  # add evaluation logic as needed
