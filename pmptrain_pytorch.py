import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import time
from torch.nn import functional as F
# Define the PMP model in PyTorch
class PMP(nn.Module):
    def __init__(self, units_layer1=128, units_layer2=128, units_layer3=128, units_layer4=128):
        super(PMP, self).__init__()
        self.dense1 = nn.Linear(6, units_layer1)
        self.dense2 = nn.Linear(units_layer1, units_layer2)
        self.dense3 = nn.Linear(units_layer2, units_layer3)
        self.dense4 = nn.Linear(units_layer3, units_layer4)
        self.dense5 = nn.Linear(units_layer4, 3)

    def forward(self, x):
        x = F.silu(self.dense1(x))
        x = F.silu(self.dense2(x))
        x = F.silu(self.dense3(x))
        x = F.silu(self.dense4(x))
        return self.dense5(x)

    def get_jacobian(self, inputs):
        inputs = inputs.requires_grad_()
        outputs = self.forward(inputs)
        jacobian = []
        for i in range(outputs.shape[1]):
            grad_outputs = torch.zeros_like(outputs)
            grad_outputs[:, i] = 1.0
            jacobian.append(torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs, retain_graph=True)[0])
        return torch.stack(jacobian, dim=1)

# Function to load data
def load_data(file_path, num_rows, num_cols):
    data = np.zeros((num_rows, num_cols), dtype=np.float32)
    try:
        with open(file_path, 'r') as f:
            matrix = f.read().split()
            for i in range(num_rows):
                for j in range(num_cols):
                    data[i, j] = float(matrix[i * num_cols + j])
    except FileNotFoundError:
        print(f"Oops! Cannot find the file {file_path}.")
    return data

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = PMP().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load data
num_data = 500000
x = load_data('IndhRobot_ur5.txt', num_data, 6)
y = load_data('OutdhRobot_ur5.txt', num_data, 3)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

# Training loop
batch_size = 128
epochs = 2000
num_batches = len(x_train) // batch_size
losses = []  # Track all training losses
val_losses = []  # Track all validation losses

# Early stopping setup
best_val_loss = float('inf')
patience = 100
wait = 0

# Measure training time
start_time = time.time()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    # Shuffle and batch data
    permutation = torch.randperm(x_train.size(0))
    for i in range(0, x_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = x_train[indices], y_train[indices]

        # Forward pass
        outputs = model(batch_x)
        loss = loss_fn(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val)
        val_loss = loss_fn(val_outputs, y_val).item()

    # Scheduler step
    scheduler.step()

    # Save losses for history
    losses.append(epoch_loss / num_batches)
    val_losses.append(val_loss)

    # Log progress
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: loss = {epoch_loss / num_batches:.6f}, val_loss = {val_loss:.6f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# Calculate and print the total training time
end_time = time.time()
training_time_seconds = end_time - start_time
training_time_hours = training_time_seconds / 3600
print(f"Total training time: {training_time_hours:.2f} hours")

# Save the model
torch.save(model.state_dict(), 'best_model.pth')

# Save training history
with open('losses.txt', 'w') as f:
    for epoch, (loss, val_loss) in enumerate(zip(losses, val_losses), start=1):
        f.write(f"Epoch {epoch}: Loss: {loss:.6f}, Val Loss: {val_loss:.6f}\n")
