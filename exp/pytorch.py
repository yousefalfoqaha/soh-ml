import numpy as np
import torch

m = 5
rng = np.random.default_rng(seed=69)
X = rng.random((m, 1))
y = 2 + 5 * X + rng.standard_normal((m, 1))

X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

torch.manual_seed(69)
w = torch.randn((1, 1), requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.2
n_epochs = 20

for epoch in range(n_epochs):
    y_pred = X_train @ w + b
    mse_loss = ((y_pred - y_train) ** 2).mean()
    mse_loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {mse_loss.item()}")
