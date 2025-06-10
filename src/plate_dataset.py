import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
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
        annotations_dir (Path): Directory containing the plate annotations
        transform (transforms.Compose): Transformations to apply to the images
        image_size (Tuple[int, int]): Target size for the images
        classes (Dict[str, int]): Mapping of class names to numeric labels
        data (List[Dict]): List of data samples with image paths and annotations
    """
    
    def __init__(
        self,
        image_dir: str,
        annotations_dir: List[Dict],
        image_size: Tuple[int, int] = (32, 32),
        transform: Optional[transforms.Compose] = None
    ):
        """Initialize the PlateDataset.
        
        Args:
            image_dir: Directory containing the plate images
            annotations_dir: Directory containing the plate annotations
            image_size: Target size for the images (height, width)
            transform: Optional transforms to apply to the images
        """
        self.image_dir = Path(image_dir)
        self.annotations_dir = annotations_dir
        self.image_size = image_size
        
        # Create class mapping
        self.classes = {"big_vrac": 0, "small_vrac": 1}
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transform
            
        # Prepare data samples
        self.data = self._prepare_data()
        
        logger.info(f"Loaded {len(self.data)} plate images from {image_dir}")
        
    def _prepare_data(self) -> List[Dict]:
        """Process annotations and prepare data samples.
        
        Returns:
            List of dictionaries containing image paths and annotations
        """
        data = []

        # Open and read the JSON file
        with open(self.annotations_dir, 'r') as file:
            annotations = json.load(file)
        
        for img_info in annotations:
            img_path = self.image_dir / img_info['name']
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
                
            # Process each box in the image
            for box_info in img_info['boxes']:
                if box_info['is_background']:
                    continue
                    
                # Convert box format from [x, y, width, height] to [x1, y1, x2, y2]
                bbox = [
                    box_info['box'][0],  # x1
                    box_info['box'][1],  # y1
                    box_info['box'][0] + box_info['box'][2],  # x2
                    box_info['box'][1] + box_info['box'][3]   # y2
                ]
                
                # Get class label
                class_label = self.classes[box_info['id']]
                
                data.append({
                    'image_path': img_path,
                    'bbox': bbox,
                    'class': class_label,
                    'box_id': box_info['box_id']
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
        image = image.crop(bbox)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return image, sample['class']
        
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class labels to their counts
        """
        class_counts = {0: 0, 1: 0}  # Initialize for both classes
        
        for sample in self.data:
            class_label = sample['class']
            class_counts[class_label] += 1
            
        return class_counts
