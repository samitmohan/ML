# https://www.deep-ml.com/labs/13?returnTo=paths

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    N = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()

        # ---- Shuffle training data ----
        perm = torch.randperm(N)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        total_train_loss = 0.0

        # ---- Mini-batch training ----
        for i in range(0, N, batch_size):
            x_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            optimizer.zero_grad()

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x_batch.size(0)

        avg_train_loss = total_train_loss / N

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()

            preds = val_logits.argmax(dim=1)
            val_accuracy = (preds == y_val).float().mean().item()

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    return history

