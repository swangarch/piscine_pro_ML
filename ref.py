
import torch.nn as nn
import torch.nn.functional as F

...,  # 2D convolutional layer with 20 filters, each 3x3, using ReLU activation, input shape is 28x28x1 (image dimensions)
...,  # Max-pooling layer with a 2x2 window to reduce image size
...,  # 2D convolutional layer with 32 filters, each 3x3, using ReLU activation
...,  # Max-pooling layer with a 2x2 window
...,  # 2D convolutional layer with 32 filters, each 2x2, using ReLU activation
...,  # Max-pooling layer with a 2x2 window
...,  # Flatten layer to convert the 2D output to 1D
...,  # Fully connected layer with 200 neurons and ReLU activation
...,  # Dropout layer with a dropout rate of 0.2 for regularization to prevent overfitting
...  # Fully connected layer with 10 output units (classes) and sigmoid activation

class AdvancdeModel(nn.Module):
  def __init__(self):
    super(AdvancdeModel, self).__init__()
    self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
    self.conv2 = nn.Conv2d(20, 32, kernel_size=3)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=2)
    self.fc1 = nn.Linear(128, 200) # Corrected input size to 32*2*2
    self.fc2 = nn.Linear(200, 10)
    #self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv3(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 128)
    x = F.relu(self.fc1(x))
    #x = self.dropout(x)
    x = self.fc2(x)
    return x
    
    
=============================================
model_torch = AdvancdeModel()
model_torch



=============================================
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_torch.parameters(), lr=0.001)

===============================================
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).squeeze()
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).squeeze()
X_test_tensor = torch.tensor(X_test[0], dtype=torch.float32).unsqueeze(1)

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


======================================================
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")
early_stopping = EarlyStopping(patience=35, delta=0.1, verbose=True)

=============================================================
# Train script
num_epochs = 10
train_loss = 0.0
val_loss = 0.0
train_loss_per_epoch = []
val_loss_per_epoch = []
for epoch in range(num_epochs):
    model_torch.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_torch(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_loss_per_epoch.append(loss.item())

    model_torch.eval()
    with torch.no_grad():
        total_val_loss = 0
        for images, labels in val_loader:
            outputs = model_torch(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_per_epoch.append(avg_val_loss)

    val_loss /= len(val_loader)
    early_stopping.check_early_stop(val_loss)

    if early_stopping.stop_training:
        print(f"Early stopping at epoch {epoch}")
        break

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_per_epoch[-1]:.4f}, Val Loss: {val_loss_per_epoch[-1]:.4f}")
    
    =========================================================
    # Predict on the test set
model_torch.eval()  # Set the model to evaluation mode
predictions_torch = []
with torch.no_grad():  # Disable gradient calculation for inference
    for images in test_loader:
        outputs = model_torch(images[0])  # Access images from the tuple
        _, predicted = torch.max(outputs.data, 1)
        predictions_torch.extend(predicted.tolist())
=======================================
# Show the structure of the tensors in the test_dataset
image_show(test_df, 42, visualize=True, test_df=True)
print(predictions_torch[42])

image_show(test_df, 424, visualize=True, test_df=True)
print(predictions_torch[424])
    
    