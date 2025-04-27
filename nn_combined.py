"""
Runs classification on combined data
"""

df_train1 = pd.read_csv(r"data/processed/train.csv")
df_test = pd.read_csv(r"data/processed/test.csv")
df_train2 = pd.read_csv(r"data/generated/synthetic_patients_with_labels.csv")

df_train2_mi = df_train2[df_train2['cvd_flag'] == 1] #keep only the MIs
#combine the datasets together
df_train = pd.concat([df_train1, df_train2_mi], ignore_index=True)
#Drop the 'source' column from combined train set
df_train = df_train.drop(columns=['source'])

# Separate predictors and target variable; drop some columns that will not be use.
X = df_train.drop(columns=["PATIENT", "BIRTHDATE", "BIRTHDATE_ORD", "DEATHDATE", "cvd_flag"])
Y = df_train["cvd_flag"].astype(float)

X_test = df_test.drop(columns=["PATIENT", "BIRTHDATE", "BIRTHDATE_ORD", "DEATHDATE", "cvd_flag"])
Y_test = df_test["cvd_flag"].astype(float)

# Convert any object columns to numeric
object_cols = X.select_dtypes(include=['object']).columns
for col in object_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

object_cols = X_test.select_dtypes(include=['object']).columns
for col in object_cols:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

X.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# Convert the feature DataFrames to PyTorch tensors.
X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

#############################################
# Using simple feed-forward neural network
#############################################
# Set up DataLoaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
class HeartFailureModel(nn.Module):
    def __init__(self, input_dim):
        super(HeartFailureModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Model setup
input_dim = X_train_tensor.shape[1]
model = HeartFailureModel(input_dim)

# Calculate pos_weight
pos_weight_value = (9174 / 526)
pos_weight = torch.tensor([pos_weight_value])

# Loss function with weighting
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 20
loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_features, batch_labels in train_loader:
        outputs = model(batch_features).squeeze(1)
        batch_labels = batch_labels.squeeze(1)

        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_features.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    loss_history.append(epoch_loss)

# Evaluate the model
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        outputs = model(batch_features).squeeze(1)
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.12).float()

        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

        all_preds.append(predicted.cpu())
        all_labels.append(batch_labels.cpu())

accuracy = correct / total
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

# Print confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)
