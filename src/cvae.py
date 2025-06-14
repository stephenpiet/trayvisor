import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        bottleneck_dim: int,
        num_classes: int,
        network_type: str = "cnn",
        class_embedding_dim: int = 64,
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.network_type = network_type

        # Shared class embedding layer
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)

        if network_type == "cnn":
            h, w = input_shape[1], input_shape[2]
            self.encoder_flat_dim = 64 * (h // 4) * (w // 4)

            self.encoder = CNNEncoder(
                input_shape,
                bottleneck_dim * 2,  # mu and log_sigma
                num_classes,
            )
            self.decoder = CNNDecoder(
                input_shape, bottleneck_dim, num_classes, self.encoder_flat_dim
            )
        else:
            raise ValueError(f"Unsupported network type: {network_type}")

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def encode(self, x, condition):
        """Encode input with conditioning. Returns exactly (mu, log_sigma)"""
        latent_params = self.encoder(x, condition)
        mu, log_sigma = torch.chunk(latent_params, 2, dim=1)
        return mu, log_sigma

    def decode(self, latent_code, condition):
        """Decode with conditioning."""
        return self.decoder(latent_code, condition)

    def bottleneck(self, mu, log_sigma):
        """Sample from latent space."""
        std = log_sigma.exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def classify_from_latent(self, z):
        """Classify from latent space representation."""
        return self.classifier(z)

    def forward(self, x, condition):
        """Full forward pass. Returns (recon, mu, log_sigma)"""
        mu, log_sigma = self.encode(x, condition)
        z = self.bottleneck(mu, log_sigma)
        recon = self.decode(z, condition)
        return recon, mu, log_sigma

    def get_class_embeddings(self):
        """Get raw class embeddings."""
        return self.class_embedding.weight

    def generate(self, condition, num_samples=1):
        """Generate samples for given conditions."""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            if isinstance(condition, int):
                condition = torch.tensor([condition], device=device)
            elif isinstance(condition, list):
                condition = torch.tensor(condition, device=device)

            # Create random latent codes
            z = torch.randn(len(condition), self.bottleneck_dim, device=device)

            # Generate samples
            return self.decode(z, condition)

    def generate_by_class(self, class_idx, num_samples=1):
        """Generate samples for a specific class."""
        return self.generate(
            torch.full((num_samples,), class_idx, device=next(self.parameters()).device)
        )

    def sample(self, num_samples=1, condition=None):
        """Generate samples (compatibility with trainer)."""
        if condition is None:
            condition = torch.randint(
                0,
                self.num_classes,
                (num_samples,),
                device=next(self.parameters()).device,
            )
        return self.generate(condition, num_samples)


class CNNEncoder(nn.Module):
    def __init__(self, input_shape, output_dim, num_classes):
        super().__init__()
        in_channels, h, w = input_shape
        self.h4 = h // 4
        self.w4 = w // 4

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        self.bn1 = ClassConditionalBatchNorm2d(16, num_classes)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = ClassConditionalBatchNorm2d(32, num_classes)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = ClassConditionalBatchNorm2d(64, num_classes)

        self.fc = nn.Linear(64 * self.h4 * self.w4, output_dim)

    def forward(self, x, condition):
        x = F.relu(self.bn1(self.conv1(x), condition))
        x = F.relu(self.bn2(self.conv2(x), condition))
        x = F.relu(self.bn3(self.conv3(x), condition))
        x = x.flatten(1)
        return self.fc(x)


class CNNDecoder(nn.Module):
    def __init__(self, input_shape, bottleneck_dim, num_classes, encoder_flat_dim):
        super().__init__()
        in_channels, h, w = input_shape
        self.h4 = h // 4
        self.w4 = w // 4

        self.fc = nn.Linear(bottleneck_dim, encoder_flat_dim)

        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn1 = ClassConditionalBatchNorm2d(32, num_classes)
        self.deconv2 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn2 = ClassConditionalBatchNorm2d(16, num_classes)
        self.conv_out = nn.Conv2d(16, in_channels, kernel_size=5, padding=2)

    def forward(self, z, condition):
        x = self.fc(z)
        x = x.view(-1, 64, self.h4, self.w4)
        x = F.relu(self.bn1(self.deconv1(x), condition))
        x = F.relu(self.bn2(self.deconv2(x), condition))
        return torch.tanh(self.conv_out(x))


class ClassConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, eps=1e-5):
        super().__init__()
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps=eps) for _ in range(num_classes)]
        )

    def forward(self, x, condition):
        out = torch.zeros_like(x)
        for class_idx in range(len(self.bn_layers)):
            mask = condition == class_idx
            if mask.any():
                out[mask] = self.bn_layers[class_idx](x[mask])
        return out
