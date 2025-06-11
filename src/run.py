import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import random_split

import dataset
import model
import trainer
from cvae import ConditionalVAE
from plate_dataset import PlateDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Variational Autoencoder.")

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        choices=["mnist", "plate"],
        default="mnist",
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--image_dir",
        help="Directory containing plate images (required for plate dataset)",
    )
    parser.add_argument(
        "--annotations_file",
        help="Path to plate annotations JSON file (required for plate dataset)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[32, 32],
        help="Target size for images (height width)",
    )

    # Model arguments
    parser.add_argument(
        "-t",
        "--network_type",
        required=True,
        choices=["mlp", "cnn"],
        help="Type of the VAE network",
    )
    parser.add_argument(
        "-n",
        "--bottleneck_dim",
        type=int,
        default=16,
        help="Size of the VAE bottleneck",
    )
    parser.add_argument(
        "--conditional", action="store_true", help="Whether to use Conditional VAE"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        help="Number of classes for Conditional VAE (required if --conditional is set)",
    )

    # Training arguments
    parser.add_argument(
        "-b", "--batch_size", type=int, required=True, help="Batch size for training"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, required=True, help="Number of epochs to train"
    )
    parser.add_argument(
        "-r", "--lr", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=20,
        help="Number of epochs to wait before early stopping",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Proportion of data to use for training",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Beta coef for KL loss",
    )

    # Hardware arguments
    parser.add_argument(
        "-d",
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to train on (e.g. "cuda:0" or "cpu")',
    )

    # Logging arguments
    parser.add_argument(
        "-l",
        "--logdir",
        default="./results",
        help="Directory to log the models and event file to",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(args, input_shape: tuple) -> nn.Module:
    """Create the VAE model.

    Args:
        args: Command line arguments
        input_shape: Shape of input images (channels, height, width)

    Returns:
        The VAE model
    """
    if args.conditional:
        if not args.num_classes:
            raise ValueError("--num_classes is required for Conditional VAE")

        logger.info(f"Creating Conditional VAE with {args.num_classes} classes")
        return ConditionalVAE(
            input_shape=input_shape,
            bottleneck_dim=args.bottleneck_dim,
            num_classes=args.num_classes,
            network_type=args.network_type,
        )
    else:
        if args.network_type == "mlp":
            return model.MLPVAE(input_shape, args.bottleneck_dim)
        elif args.network_type == "cnn":
            return model.CNNVAE(input_shape, args.bottleneck_dim)
        else:
            raise ValueError(f"Unsupported network type {args.network_type}")


def load_dataset(args):
    """Load the specified dataset.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if args.dataset == "mnist":
        logger.info("Loading MNIST dataset...")
        mnist_data = dataset.MyMNIST()
        return mnist_data.train_data, mnist_data.test_data

    elif args.dataset == "plate":
        if not args.image_dir or not args.annotations_file:
            raise ValueError(
                "--image_dir and --annotations_file are required for plate dataset"
            )

        logger.info("Loading plate dataset...\n")
        full_dataset = PlateDataset(
            image_dir=args.image_dir,
            annotations_dir=args.annotations_file,
            image_size=tuple(args.image_size),
        )

        # Log class distribution
        logger.info("Class Distribution:")
        for class_name, count in sorted(full_dataset.class_distribution.items()):
            logger.info(f"{class_name}: {count} samples")
        logger.info(f"Total samples: {len(full_dataset.data)}\n")

        # Split dataset into train and testtrain_size = int(len(full_dataset) * args.train_split)
        train_size = int(len(full_dataset) * args.train_split)
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        # Log dataset information
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

    return train_dataset, test_dataset


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create log directory
    log_dir = Path(args.logdir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    try:
        # Load dataset
        train_dataset, test_dataset = load_dataset(args)

        # Get input shape from first sample
        sample_image, _ = train_dataset[0]
        input_shape = sample_image.shape
        logger.info(f"Input shape: {input_shape}")

        # Create model
        net = create_model(args, input_shape)

        # Create trainer
        vae_trainer = trainer.Trainer(
            model=net,
            train_data=train_dataset,
            test_data=test_dataset,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            device=device,
            log_dir=log_dir,
            early_stopping_patience=args.early_stopping_patience,
            is_conditional=args.conditional,
            beta=args.beta,
        )

        # Train model
        logger.info("Starting training...")
        history = vae_trainer.train(args.epochs)

        # Log final results
        logger.info("Training completed!")
        logger.info(f"Best model saved at: {history['best_model_path']}")
        logger.info(f"Best epoch: {history['best_epoch']}")
        logger.info(
            f"Best validation loss: {history['val_loss'][history['best_epoch'] - 1]:.4f}"
        )

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
