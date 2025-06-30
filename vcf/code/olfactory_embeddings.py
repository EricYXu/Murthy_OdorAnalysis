import torch
import torch.nn as nn
import pandas as pd

"""
olfactory_embeddings.py

Script that creates a multilayer perceptron to classify olfactory mixture vectors (8730-dim) to one of 13 classes.
We will combine the "ALCOHOLIC1" and "ALCOHOLIC2" classes.

Usage:
    python3 olfactory_embeddings.py

"""


class MLP(nn.Module):
    def __init__(self, input_dim=8730, hidden_dim=512, output_dim=13, num_hidden=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_embedding(self, x):
        # Forward through all but the last layer
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return x

if __name__ == '__main__':
    # Load Matrix.csv, skip first row and first column
    raw = pd.read_csv('../Matrix.csv', header=None)
    X = raw.iloc[1:, 1:].astype(float).values
    X = X.T  # Now shape: (samples, features)
    print(f"Input shape: {X.shape}")
    assert X.shape[1] == 8730, f"Expected 8730 features, got {X.shape[1]}"

    # Convert to torch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Dummy labels (replace with real labels if available)
    n_samples = X_tensor.shape[0]
    n_classes = 13
    # For demonstration, use random labels
    y_tensor = torch.randint(0, n_classes, (n_samples,))

    # Create model
    model = MLP(input_dim=8730, hidden_dim=512, output_dim=13, num_hidden=4)
    print(model)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 10
    batch_size = 32

    for epoch in range(n_epochs):
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_tensor[indices]
            batch_y = y_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        avg_loss = epoch_loss / n_samples
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # Dummy forward pass after training
    out = model(X_tensor)
    print(f"Output shape: {out.shape}") 