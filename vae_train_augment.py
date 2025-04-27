"""
VAE augmented
"""

# Load and preprocess data
train = pd.read_csv(r"data/processed/train.csv")
drop_cols = ['PATIENT', 'BIRTHDATE', 'DEATHDATE', 'CITY', 'STATE', 'COUNTY', 'cvd_flag']
train_features = train.drop(columns=drop_cols)

# Convert object columns to numeric
object_cols = train_features.select_dtypes(include=['object']).columns
for col in object_cols:
    train_features[col] = pd.to_numeric(train_features[col], errors='coerce')
train_features = train_features.fillna(0)

# Normalize features
scaler = MinMaxScaler()
train_features_scaled = scaler.fit_transform(train_features)
# Create PyTorch tensor and DataLoader
X_tensor = torch.tensor(train_features_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
# Define Encoder with Dropout
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.output_layer = nn.Linear(256, output_dim)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        return torch.sigmoid(self.output_layer(z))

# Define VAE
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

# Define beta-VAE Loss
def vae_loss(reconstructed_x, x, mu, logvar, kl_weight=0.001):
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_weight * kl_divergence

# Model setup
input_dim = train_features.shape[1]
latent_dim = 100
vae = VAE(input_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Train the VAE
num_epochs = 50
vae.train()
loss_history = []

for epoch in range(num_epochs):
    train_loss = 0
    for batch in dataloader:
        x_batch = batch[0]
        optimizer.zero_grad()
        reconstructed_x, mu, logvar = vae(x_batch)
        loss = vae_loss(reconstructed_x, x_batch, mu, logvar, kl_weight=0.0005)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(dataloader.dataset)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
plt.title("VAE Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Generate synthetic data
n_samples = 20000
device = next(vae.parameters()).device
z = torch.randn(n_samples, latent_dim).to(device) * 2.0

with torch.no_grad():
    synthetic_data = vae.decoder(z)

synthetic_data = synthetic_data.cpu().numpy()
synthetic_data = scaler.inverse_transform(synthetic_data)

synthetic_df = pd.DataFrame(synthetic_data, columns=train_features.columns)
synthetic_df['source'] = 'synthetic'
#synthetic_df.to_csv("data/generated/synthetic_patients.csv", index=False)

# Combine real and synthetic for visualization
real_df = train_features.copy()
real_df['source'] = 'real'
combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)

# Apply PCA
pca = PCA(n_components=2)
X_combined = combined_df.drop(columns=['source'])
X_pca = pca.fit_transform(X_combined)

pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Source': combined_df['source']
})

# PCA plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Source', alpha=0.5)
plt.title("PCA Projection: Real vs Synthetic Data")
plt.grid(True)
plt.show()

#keeping these as sufficient patients, let's add cvd_flag to these patients:
# Load real train data
train_real = pd.read_csv("data/processed/train.csv")

# Features and target
drop_cols = ['PATIENT', 'BIRTHDATE', 'DEATHDATE', 'CITY', 'STATE', 'COUNTY', 'cvd_flag']
X_real = train_real.drop(columns=drop_cols)
y_real = train_real['cvd_flag'].astype(float)
# Coerce object columns to numeric
object_cols = X_real.select_dtypes(include=['object']).columns
for col in object_cols:
    X_real[col] = pd.to_numeric(X_real[col], errors='coerce')

X_real = X_real.fillna(0)

scaler = MinMaxScaler()
X_real_scaled = scaler.fit_transform(X_real)
# Convert to PyTorch tensors
X_real_tensor = torch.tensor(X_real_scaled, dtype=torch.float32)
y_real_tensor = torch.tensor(y_real.values, dtype=torch.float32).unsqueeze(1)

# Setup DataLoader
real_dataset = TensorDataset(X_real_tensor, y_real_tensor)
real_loader = DataLoader(real_dataset, batch_size=128, shuffle=True)

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

input_dim = X_real_tensor.shape[1]
classifier = SimpleClassifier(input_dim)

criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    classifier.train()
    running_loss = 0.0
    for batch_X, batch_y in real_loader:
        optimizer.zero_grad()
        outputs = classifier(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    epoch_loss = running_loss / len(real_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

#didn't output - it's too big anyway
#synthetic_df = pd.read_csv("data/generated/synthetic_patients.csv")

X_synthetic = synthetic_df[X_real.columns]

object_cols = X_synthetic.select_dtypes(include=['object']).columns
for col in object_cols:
    X_synthetic[col] = pd.to_numeric(X_synthetic[col], errors='coerce')
X_synthetic = X_synthetic.fillna(0)
X_synthetic_scaled = scaler.transform(X_synthetic)
X_synthetic_tensor = torch.tensor(X_synthetic_scaled, dtype=torch.float32)

# Predict
classifier.eval()
with torch.no_grad():
    synthetic_preds = classifier(X_synthetic_tensor)

# Threshold
synthetic_labels = (synthetic_preds > 0.5).float().squeeze().numpy()
# Add to dataframe
synthetic_df['cvd_flag'] = synthetic_labels
# Save
synthetic_df.to_csv("data/generated/synthetic_patients_with_labels.csv", index=False)

