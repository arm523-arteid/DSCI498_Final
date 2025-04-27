"""
Initial VAE train
"""

train = pd.read_csv(r"data/processed/train.csv")
# Drop non-feature columns
drop_cols = ['PATIENT', 'BIRTHDATE', 'DEATHDATE', 'CITY', 'STATE', 'COUNTY', 'cvd_flag']
train_features = train.drop(columns=drop_cols)
# After converting to numeric
object_cols = train_features.select_dtypes(include=['object']).columns
for col in object_cols:
    train_features[col] = pd.to_numeric(train_features[col], errors='coerce')
train_features = train_features.fillna(0)
# Normalize features
scaler = MinMaxScaler()
train_features_scaled = scaler.fit_transform(train_features)
# Create PyTorch tensor from SCALED features
X_tensor = torch.tensor(train_features_scaled, dtype=torch.float32)
# Create Dataset and DataLoader
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

#Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar
#Define the decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.output_layer = nn.Linear(256, output_dim)
    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        x_reconstructed = torch.sigmoid(self.output_layer(z))
        return x_reconstructed

#define the VAE
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
#Define VAE Loss
def vae_loss(reconstructed_x, x, mu, logvar):
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence
# Convert to PyTorch tensor
X_tensor = torch.tensor(train_features_scaled, dtype=torch.float32)
# Create dataset and dataloader
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
# Model setup
input_dim = train_features.shape[1]
latent_dim = 20
vae = VAE(input_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
# Train the model
num_epochs = 30
vae.train()
loss_history = []
for epoch in range(num_epochs):
    train_loss = 0
    for batch in dataloader:
        x_batch = batch[0]
        optimizer.zero_grad()
        reconstructed_x, mu, logvar = vae(x_batch)
        loss = vae_loss(reconstructed_x, x_batch, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(dataloader.dataset)
    loss_history.append(avg_loss)
    #keep track of whats going on
    #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Plotting the loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
plt.title("VAE Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Set number of synthetic patients
n_samples = 20000
# Sample random from standard normal distribution
device = next(vae.parameters()).device
z = torch.randn(n_samples, latent_dim).to(device)
#Decode to create synthetic data
with torch.no_grad():
    synthetic_data = vae.decoder(z)
#Bring synthetic data from tensors to numpy
synthetic_data = synthetic_data.cpu().numpy()
#inverse_transform it
synthetic_data = scaler.inverse_transform(synthetic_data)

#Save as a DataFrame & output to csv
synthetic_df = pd.DataFrame(synthetic_data, columns=train_features.columns)
synthetic_df['source'] = 'synthetic'
#synthetic_df.to_csv("data/generated/synthetic_patients.csv", index=False)

#bring in train features for real data to be combined with synthetic
real_df = train_features.copy()
real_df['source'] = 'real'

combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
#confirm how many are in each
#print(combined_df['source'].value_counts())

# Apply PCA
pca = PCA(n_components=2)
X_combined = combined_df.drop(columns=['source'])
X_pca = pca.fit_transform(X_combined)

# Create a DataFrame to plot
pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Source': combined_df['source']
})

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Source', alpha=0.5)
plt.title("PCA projection: Real vs Synthetic")
plt.grid(True)
plt.show()