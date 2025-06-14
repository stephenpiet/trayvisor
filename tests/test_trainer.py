import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.stats
import torch
from PIL import Image
from torch.utils.data import random_split

from src.model import CNNVAE
from src.plate_dataset import PlateDataset
from src.trainer import Trainer


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logging."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with test images."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test images
        for i in range(6):  # Create more images for train/test split
            img = Image.new("RGB", (100, 100), color="red")
            img.save(os.path.join(temp_dir, f"test_image_{i}.jpg"))
        yield temp_dir


@pytest.fixture
def temp_annotations_file():
    """Create a temporary annotations file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        annotations = [
            {
                "name": f"test_image_{i}.jpg",
                "boxes": [
                    {
                        "box": [10, 10, 30, 30],
                        "id": "big_vrac",
                        "is_background": False,
                        "box_id": i,
                    }
                ],
            }
            for i in range(6)  # Create annotations for all images
        ]
        json.dump(annotations, f)
        return f.name


@pytest.fixture
def plate_dataset(temp_image_dir, temp_annotations_file):
    """Create a small plate dataset for testing."""
    dataset = PlateDataset(
        image_dir=temp_image_dir,
        annotations_dir=temp_annotations_file,
        image_size=(32, 32),
    )
    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


@pytest.fixture
def trainer(temp_log_dir, plate_dataset):
    """Create a trainer instance for testing."""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    train_dataset, test_dataset = plate_dataset
    vae = CNNVAE(train_dataset[0][0].shape, bottleneck_dim=10)
    optim = torch.optim.Adam(vae.parameters())

    return Trainer(
        model=vae,
        train_data=train_dataset,
        test_data=test_dataset,
        batch_size=4,
        learning_rate=0.001,
        device="cpu",
        log_dir=temp_log_dir,
        num_generated_images=1,
    )


def test_kl_divergence(trainer):
    """Test KL divergence calculation."""
    mu = np.random.randn(10) * 0.25
    sigma = np.random.randn(10) * 0.1 + 1.0
    standard_normal_samples = np.random.randn(100000, 10)
    transformed_normal_sample = standard_normal_samples * sigma + mu

    # Calculate empirical pdfs for both distributions
    bins = 1000
    bin_range = [-2, 2]
    expected_kl_div = 0
    for i in range(10):
        standard_normal_dist, _ = np.histogram(
            standard_normal_samples[:, i], bins, bin_range
        )
        transformed_normal_dist, _ = np.histogram(
            transformed_normal_sample[:, i], bins, bin_range
        )
        expected_kl_div += scipy.stats.entropy(
            transformed_normal_dist, standard_normal_dist
        )

    actual_kl_div = trainer._kl_divergence(torch.tensor(sigma).log(), torch.tensor(mu))

    assert abs(expected_kl_div - actual_kl_div.numpy()) <= 0.05


def test_model_saving(trainer, temp_log_dir):
    """Test model checkpoint saving."""
    # Train for a few steps
    trainer.train(10)

    # Get the run directory (it will be a timestamp directory inside temp_log_dir)
    run_dirs = list(Path(temp_log_dir).glob("*"))
    assert len(run_dirs) == 1, "Expected exactly one run directory"
    run_dir = run_dirs[0]

    # Check if checkpoint files exist
    checkpoint_files = list(run_dir.glob("checkpoint_epoch_*.pth"))
    assert len(checkpoint_files) > 0, "No checkpoint files found"

    # Check if best model exists
    best_model = run_dir / "best_model.pth"
    assert best_model.exists(), f"Best model {best_model} does not exist"


def test_early_stopping(trainer):
    """Test early stopping functionality."""
    # Set a very small patience
    trainer.early_stopping_patience = 2

    # Train for a few steps
    trainer.train(10)

    # Check if training stopped early
    assert trainer.best_loss is not None
