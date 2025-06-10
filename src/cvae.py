import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder.
    
    This model extends the standard VAE by conditioning both the encoder and decoder
    on additional information (e.g., class labels). The model learns to generate
    samples that match the given conditions.
    
    Attributes:
        bottleneck_dim (int): Dimension of the latent space
        num_classes (int): Number of possible conditions
        encoder (nn.Module): Encoder network
        decoder (nn.Module): Decoder network
    """
    
    def __init__(
        self,
        input_shape: tuple,
        bottleneck_dim: int,
        num_classes: int,
        network_type: str = 'cnn'
    ):
        """Initialize the Conditional VAE.
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            bottleneck_dim: Dimension of the latent space
            num_classes: Number of possible conditions
            network_type: Type of network architecture ('cnn' or 'mlp')
        """
        super(ConditionalVAE, self).__init__()
        
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # Create encoder and decoder based on network type
        if network_type == 'cnn':
            self.encoder = CNNEncoder(input_shape, bottleneck_dim, num_classes)
            self.decoder = CNNDecoder(input_shape, bottleneck_dim, num_classes)
        elif network_type == 'mlp':
            self.encoder = MLPEncoder(input_shape, bottleneck_dim, num_classes)
            self.decoder = MLPDecoder(input_shape, bottleneck_dim, num_classes)
        else:
            raise ValueError(f"Unsupported network type: {network_type}")
            
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            condition: Condition tensor (e.g., class labels)
            
        Returns:
            Reconstructed output
        """
        mu, log_sigma = self.encode(x, condition)
        latent_code = self.bottleneck(mu, log_sigma)
        return self.decode(latent_code, condition)
        
    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> tuple:
        """Encode input and condition to latent parameters.
        
        Args:
            x: Input tensor
            condition: Condition tensor
            
        Returns:
            Tuple of (mu, log_sigma) for the latent distribution
        """
        latent_parameters = self.encoder(x, condition)
        mu, log_sigma = torch.split(latent_parameters, self.bottleneck_dim, dim=1)
        return mu, log_sigma
        
    def bottleneck(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """Sample from the latent distribution.
        
        Args:
            mu: Mean of the latent distribution
            log_sigma: Log standard deviation of the latent distribution
            
        Returns:
            Sampled latent code
        """
        noise = torch.randn_like(mu)
        return log_sigma.exp() * noise + mu
        
    def decode(self, latent_code: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode latent code and condition to output.
        
        Args:
            latent_code: Latent code tensor
            condition: Condition tensor
            
        Returns:
            Reconstructed output
        """
        return self.decoder(latent_code, condition)
        
    def generate(self, condition: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Generate samples for the given condition.
        
        Args:
            condition: Condition tensor
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        self.eval()
        with torch.no_grad():
            # Sample from standard normal distribution
            latent_code = torch.randn(
                num_samples,
                self.bottleneck_dim,
                device=condition.device
            )
            # Generate samples
            return self.decode(latent_code, condition)


class CNNEncoder(nn.Module):
    """CNN-based encoder for the Conditional VAE."""
    
    def __init__(self, input_shape: tuple, bottleneck_dim: int, num_classes: int):
        """Initialize the CNN encoder.
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            bottleneck_dim: Dimension of the latent space
            num_classes: Number of possible conditions
        """
        super(CNNEncoder, self).__init__()
        
        in_channels = input_shape[0]
        hw = input_shape[1]
        hw_before_linear = hw // 4
        flat_dim = 64 * hw_before_linear ** 2
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(num_classes, 32)
        
        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten()
        )
        
        # Combine features and condition
        self.fc = nn.Sequential(
            nn.Linear(flat_dim + 32, 512),
            nn.ReLU(True),
            nn.Linear(512, 2 * bottleneck_dim)
        )
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.
        
        Args:
            x: Input tensor
            condition: Condition tensor
            
        Returns:
            Latent parameters
        """
        # Process input through CNN
        features = self.conv_layers(x)
        
        # Process condition
        condition_embed = self.condition_embedding(condition)
        
        # Combine features and condition
        combined = torch.cat([features, condition_embed], dim=1)
        
        # Generate latent parameters
        return self.fc(combined)


class CNNDecoder(nn.Module):
    """CNN-based decoder for the Conditional VAE."""
    
    def __init__(self, input_shape: tuple, bottleneck_dim: int, num_classes: int):
        """Initialize the CNN decoder.
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            bottleneck_dim: Dimension of the latent space
            num_classes: Number of possible conditions
        """
        super(CNNDecoder, self).__init__()
        
        in_channels = input_shape[0]
        hw = input_shape[1]
        hw_before_linear = hw // 4
        flat_dim = 64 * hw_before_linear ** 2
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(num_classes, 32)
        
        # Initial fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(bottleneck_dim + 32, flat_dim),
            nn.ReLU(True)
        )
        
        # Transposed convolution layers
        self.deconv_layers = nn.Sequential(
            Unflatten((64, hw_before_linear, hw_before_linear)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, in_channels, kernel_size=5, padding=2),
            nn.Tanh()
        )
        
    def forward(self, latent_code: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.
        
        Args:
            latent_code: Latent code tensor
            condition: Condition tensor
            
        Returns:
            Reconstructed output
        """
        # Process condition
        condition_embed = self.condition_embedding(condition)
        
        # Combine latent code and condition
        combined = torch.cat([latent_code, condition_embed], dim=1)
        
        # Process through fully connected layer
        x = self.fc(combined)
        
        # Process through transposed convolutions
        return self.deconv_layers(x)


class MLPEncoder(nn.Module):
    """MLP-based encoder for the Conditional VAE."""
    
    def __init__(self, input_shape: tuple, bottleneck_dim: int, num_classes: int):
        """Initialize the MLP encoder.
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            bottleneck_dim: Dimension of the latent space
            num_classes: Number of possible conditions
        """
        super(MLPEncoder, self).__init__()
        
        in_channels = input_shape[0]
        hw = input_shape[1]
        flat_dim = in_channels * (hw ** 2)
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(num_classes, 32)
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True)
        )
        
        # Combine features and condition
        self.fc = nn.Sequential(
            nn.Linear(128 + 32, 2 * bottleneck_dim)
        )
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.
        
        Args:
            x: Input tensor
            condition: Condition tensor
            
        Returns:
            Latent parameters
        """
        # Process input through MLP
        features = self.mlp(x)
        
        # Process condition
        condition_embed = self.condition_embedding(condition)
        
        # Combine features and condition
        combined = torch.cat([features, condition_embed], dim=1)
        
        # Generate latent parameters
        return self.fc(combined)


class MLPDecoder(nn.Module):
    """MLP-based decoder for the Conditional VAE."""
    
    def __init__(self, input_shape: tuple, bottleneck_dim: int, num_classes: int):
        """Initialize the MLP decoder.
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            bottleneck_dim: Dimension of the latent space
            num_classes: Number of possible conditions
        """
        super(MLPDecoder, self).__init__()
        
        in_channels = input_shape[0]
        hw = input_shape[1]
        flat_dim = in_channels * (hw ** 2)
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(num_classes, 32)
        
        # Initial fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(bottleneck_dim + 32, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, flat_dim),
            Unflatten(input_shape),
            nn.Tanh()
        )
        
    def forward(self, latent_code: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.
        
        Args:
            latent_code: Latent code tensor
            condition: Condition tensor
            
        Returns:
            Reconstructed output
        """
        # Process condition
        condition_embed = self.condition_embedding(condition)
        
        # Combine latent code and condition
        combined = torch.cat([latent_code, condition_embed], dim=1)
        
        # Process through fully connected layers
        return self.fc(combined)


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