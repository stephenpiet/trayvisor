import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json
import logging

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlateDataset(Dataset):
    """Dataset class for handling plate images and their annotations.
    
    This dataset loads plate images and their corresponding bounding box annotations,
    crops the images to the plate regions, and applies necessary transformations.
    
    Attributes:
        image_dir (Path): Directory containing the plate images
        annotations_file (Path): Path to the JSON file containing annotations
        transform (transforms.Compose): Transformations to apply to the images
        image_size (Tuple[int, int]): Target size for the images
        data (List[Dict]): List of data samples with image paths and annotations
    """
    
    def __init__(
        self,
        image_dir: str,
        annotations_file: str,
        image_size: Tuple[int, int] = (32, 32),
        transform: Optional[transforms.Compose] = None
    ):
        """Initialize the PlateDataset.
        
        Args:
            image_dir: Directory containing the plate images
            annotations_file: Path to the JSON file containing annotations
            image_size: Target size for the images (height, width)
            transform: Optional transforms to apply to the images
        """
        self.image_dir = Path(image_dir)
        self.annotations_file = Path(annotations_file)
        self.image_size = image_size
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transform
            
        # Load annotations
        self.data = self._load_annotations()
        
        logger.info(f"Loaded {len(self.data)} plate images from {image_dir}")
        
    def _load_annotations(self) -> list:
        """Load and validate annotations from the JSON file.
        
        Returns:
            List of dictionaries containing image paths and annotations
        """
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
            
        with open(self.annotations_file, 'r') as f:
            annotations = json.load(f)
            
        data = []
        for img_info in annotations:
            img_path = self.image_dir / img_info['filename']
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
                
            # Extract bounding box and class information
            bbox = img_info['bbox']  # [x, y, width, height]
            class_label = img_info['class']
            
            data.append({
                'image_path': img_path,
                'bbox': bbox,
                'class': class_label
            })
            
        return data
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple of (image tensor, class label)
        """
        sample = self.data[idx]
        
        # Load and crop image
        image = Image.open(sample['image_path']).convert('L')  # Convert to grayscale
        bbox = sample['bbox']
        
        # Crop image to plate region
        image = image.crop((
            bbox[0],
            bbox[1],
            bbox[0] + bbox[2],
            bbox[1] + bbox[3]
        ))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return image, sample['class']
        
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class labels to their counts
        """
        class_counts = {}
        for sample in self.data:
            class_label = sample['class']
            class_counts[class_label] = class_counts.get(class_label, 0) + 1
        return class_counts 