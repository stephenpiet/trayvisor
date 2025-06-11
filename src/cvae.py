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
        class_embedding_dim: int = 32,
    ):
        super(ConditionalVAE, self).__init__()
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.class_embedding_dim = class_embedding_dim

        if network_type == "cnn":
            self.encoder = CNNEncoder(
                input_shape, bottleneck_dim, num_classes, class_embedding_dim
            )
            self.decoder = CNNDecoder(
                input_shape, bottleneck_dim, num_classes, class_embedding_dim
            )
        elif network_type == "mlp":
            self.encoder = MLPEncoder(
                input_shape, bottleneck_dim, num_classes, class_embedding_dim
            )
            self.decoder = MLPDecoder(
                input_shape, bottleneck_dim, num_classes, class_embedding_dim
            )
        else:
            raise ValueError(f"Unsupported network type: {network_type}")

    def forward(self, x, condition):
        mu, log_sigma = self.encode(x, condition)
        latent_code = self.bottleneck(mu, log_sigma)
        outputs = self.decode(latent_code, condition)
        return outputs, mu, log_sigma

    def encode(self, x, condition):
        latent_parameters = self.encoder(x, condition)
        mu, log_sigma = torch.chunk(latent_parameters, 2, dim=1)
        return mu, log_sigma

    def bottleneck(self, mu, log_sigma):
        std = log_sigma.exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent_code, condition):
        return self.decoder(latent_code, condition)

    def generate(self, condition, num_samples=1):
        self.eval()
        with torch.no_grad():
            latent_code = torch.randn(
                num_samples, self.bottleneck_dim, device=condition.device
            )
            return self.decode(latent_code, condition)

    def get_class_embeddings(self):
        return self.encoder.class_embedding.weight.data


class CNNEncoder(nn.Module):
    def __init__(self, input_shape, bottleneck_dim, num_classes, class_embedding_dim):
        super(CNNEncoder, self).__init__()
        in_channels, h, _ = input_shape
        h4 = h // 4
        flat_dim = 64 * h4 * h4

        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        self.bn1 = ClassConditionalBatchNorm2d(16, num_classes)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = ClassConditionalBatchNorm2d(32, num_classes)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = ClassConditionalBatchNorm2d(64, num_classes)

        self.fc = nn.Sequential(
            nn.Linear(flat_dim + class_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * bottleneck_dim),
        )

    def forward(self, x, condition):
        x = F.relu(self.bn1(self.conv1(x), condition))
        x = F.relu(self.bn2(self.conv2(x), condition))
        x = F.relu(self.bn3(self.conv3(x), condition))
        x = x.view(x.size(0), -1)
        cond_embed = self.class_embedding(condition)
        return self.fc(torch.cat([x, cond_embed], dim=1))


class CNNDecoder(nn.Module):
    def __init__(self, input_shape, bottleneck_dim, num_classes, class_embedding_dim):
        super(CNNDecoder, self).__init__()
        in_channels, h, _ = input_shape
        h4 = h // 4
        flat_dim = 64 * h4 * h4
        self.h4 = h4

        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(bottleneck_dim + class_embedding_dim, flat_dim),
            nn.ReLU(),
        )

        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn1 = ClassConditionalBatchNorm2d(32, num_classes)
        self.deconv2 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn2 = ClassConditionalBatchNorm2d(16, num_classes)
        self.conv_out = nn.Conv2d(16, in_channels, kernel_size=5, padding=2)

    def forward(self, latent_code, condition):
        cond_embed = self.class_embedding(condition)
        x = self.fc(torch.cat([latent_code, cond_embed], dim=1))
        x = x.view(x.size(0), 64, self.h4, self.h4)
        x = F.relu(self.bn1(self.deconv1(x), condition))
        x = F.relu(self.bn2(self.deconv2(x), condition))
        return torch.tanh(self.conv_out(x))


class MLPEncoder(nn.Module):
    def __init__(self, input_shape, bottleneck_dim, num_classes, class_embedding_dim):
        super(MLPEncoder, self).__init__()
        c, h, w = input_shape
        self.flat_dim = c * h * w
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, 512),
            ClassConditionalBatchNorm1d(512, num_classes),
            nn.ReLU(),
            nn.Linear(512, 256),
            ClassConditionalBatchNorm1d(256, num_classes),
            nn.ReLU(),
            nn.Linear(256, 128),
            ClassConditionalBatchNorm1d(128, num_classes),
            nn.ReLU(),
        )

        self.fc = nn.Linear(128 + class_embedding_dim, 2 * bottleneck_dim)

    def forward(self, x, condition):
        x = self.mlp(x)
        cond_embed = self.class_embedding(condition)
        return self.fc(torch.cat([x, cond_embed], dim=1))


class MLPDecoder(nn.Module):
    def __init__(self, input_shape, bottleneck_dim, num_classes, class_embedding_dim):
        super(MLPDecoder, self).__init__()
        c, h, w = input_shape
        flat_dim = c * h * w
        self.input_shape = input_shape
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim + class_embedding_dim, 128),
            ClassConditionalBatchNorm1d(128, num_classes),
            nn.ReLU(),
            nn.Linear(128, 256),
            ClassConditionalBatchNorm1d(256, num_classes),
            nn.ReLU(),
            nn.Linear(256, 512),
            ClassConditionalBatchNorm1d(512, num_classes),
            nn.ReLU(),
            nn.Linear(512, flat_dim),
            nn.Tanh(),
        )

    def forward(self, latent_code, condition):
        cond_embed = self.class_embedding(condition)
        x = self.mlp(torch.cat([latent_code, cond_embed], dim=1))
        return x.view(-1, *self.input_shape)


class ClassConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_classes, eps=1e-5):
        super().__init__()
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(num_features, eps=eps) for _ in range(num_classes)]
        )

    def forward(self, x, condition):
        out = torch.zeros_like(x)
        for class_idx in torch.unique(condition):
            mask = condition == class_idx
            if mask.any():
                out[mask] = self.bn_layers[class_idx](x[mask])
        return out


class ClassConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, eps=1e-5):
        super().__init__()
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps=eps) for _ in range(num_classes)]
        )

    def forward(self, x, condition):
        out = torch.zeros_like(x)
        for class_idx in torch.unique(condition):
            mask = condition == class_idx
            if mask.any():
                out[mask] = self.bn_layers[class_idx](x[mask])
        return out


class Unflatten(nn.Module):
    """Layer to reshape a flattened tensor back to its original shape."""

    def __init__(self, shape: tuple):
        """Initialize the Unflatten layer.

        Args:
            shape: Expected output shape without batch dimension
        """
        super(Unflatten, self).__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Reshaped tensor
        """
        return torch.reshape(x, (-1,) + self.shape)
