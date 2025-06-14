import json
import os
import tempfile

import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.plate_dataset import PlateDataset


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with test images."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test images
        for i in range(3):
            img = Image.new("RGB", (100, 100), color="red")
            img.save(os.path.join(temp_dir, f"test_image_{i}.jpg"))
        yield temp_dir


@pytest.fixture
def temp_annotations_file():
    """Create a temporary annotations file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        annotations = [
            {
                "name": "test_image_0.jpg",
                "boxes": [
                    {
                        "box": [10, 10, 30, 30],
                        "id": "big_vrac",
                        "is_background": False,
                        "box_id": 0,
                    }
                ],
            },
            {
                "name": "test_image_1.jpg",
                "boxes": [
                    {
                        "box": [20, 20, 40, 40],
                        "id": "small_vrac",
                        "is_background": False,
                        "box_id": 1,
                    }
                ],
            },
        ]
        json.dump(annotations, f)
        return f.name


def test_plate_dataset_initialization(temp_image_dir, temp_annotations_file):
    """Test PlateDataset initialization."""
    dataset = PlateDataset(
        image_dir=temp_image_dir,
        annotations_dir=temp_annotations_file,
        image_size=(32, 32),
    )

    assert len(dataset) == 2  # Two valid annotations
    assert dataset.classes == {"big_vrac": 0, "small_vrac": 1}


def test_plate_dataset_getitem(temp_image_dir, temp_annotations_file):
    """Test PlateDataset __getitem__ method."""
    dataset = PlateDataset(
        image_dir=temp_image_dir,
        annotations_dir=temp_annotations_file,
        image_size=(32, 32),
    )

    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)  # RGB image
    assert isinstance(label, int)
    assert label in [0, 1]


def test_plate_dataset_grayscale(temp_image_dir, temp_annotations_file):
    """Test PlateDataset with grayscale images."""
    dataset = PlateDataset(
        image_dir=temp_image_dir,
        annotations_dir=temp_annotations_file,
        image_size=(32, 32),
        gray_scale=True,
    )

    image, _ = dataset[0]
    assert image.shape == (1, 32, 32)  # Grayscale image


def test_plate_dataset_class_distribution(temp_image_dir, temp_annotations_file):
    """Test PlateDataset class distribution."""
    dataset = PlateDataset(
        image_dir=temp_image_dir,
        annotations_dir=temp_annotations_file,
        image_size=(32, 32),
    )

    distribution = dataset.get_class_distribution()
    assert "big_vrac" in distribution
    assert "small_vrac" in distribution
    assert distribution["big_vrac"] == 1
    assert distribution["small_vrac"] == 1


def test_plate_dataset_custom_transform(temp_image_dir, temp_annotations_file):
    """Test PlateDataset with custom transforms."""
    custom_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    dataset = PlateDataset(
        image_dir=temp_image_dir,
        annotations_dir=temp_annotations_file,
        image_size=(32, 32),
        transform=custom_transform,
    )

    image, _ = dataset[0]
    assert image.shape == (3, 32, 32)
