import pytest
import torch

from src.cvae import ClassConditionalBatchNorm2d, ConditionalVAE


@pytest.fixture
def cvae_model():
    return ConditionalVAE(
        input_shape=(3, 32, 32),
        bottleneck_dim=256,
        num_classes=2,
        network_type="cnn",
        class_embedding_dim=64,
    )


def test_cvae_initialization(cvae_model):
    """Test CVAE model initialization."""
    assert cvae_model.bottleneck_dim == 256
    assert cvae_model.num_classes == 2
    assert cvae_model.input_shape == (3, 32, 32)
    assert cvae_model.network_type == "cnn"


def test_cvae_forward_pass(cvae_model):
    """Test CVAE forward pass."""
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    condition = torch.randint(0, 2, (batch_size,))

    recon, mu, log_sigma = cvae_model(x, condition)

    assert recon.shape == (batch_size, 3, 32, 32)
    assert mu.shape == (batch_size, 256)
    assert log_sigma.shape == (batch_size, 256)


def test_cvae_generate(cvae_model):
    """Test CVAE sample generation."""
    num_samples = 4
    condition = torch.randint(0, 2, (num_samples,))

    samples = cvae_model.generate(condition)
    assert samples.shape == (num_samples, 3, 32, 32)


def test_cvae_generate_by_class(cvae_model):
    """Test CVAE generation for specific class."""
    num_samples = 4
    class_idx = 1

    samples = cvae_model.generate_by_class(class_idx, num_samples)
    assert samples.shape == (num_samples, 3, 32, 32)


def test_cvae_classifier(cvae_model):
    """Test CVAE classifier head."""
    batch_size = 4
    z = torch.randn(batch_size, 256)

    logits = cvae_model.classify_from_latent(z)
    assert logits.shape == (batch_size, 2)


def test_class_conditional_batch_norm():
    """Test ClassConditionalBatchNorm2d layer."""
    batch_size = 4
    num_features = 16
    num_classes = 2

    layer = ClassConditionalBatchNorm2d(num_features, num_classes)
    x = torch.randn(batch_size, num_features, 8, 8)
    condition = torch.randint(0, num_classes, (batch_size,))

    out = layer(x, condition)
    assert out.shape == x.shape
