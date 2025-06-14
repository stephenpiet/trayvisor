import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to monitor training and stop when validation loss stops improving."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, current_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            current_loss: Current validation loss
            model: Model being trained

        Returns:
            True if training should stop, False otherwise
        """
        if (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            if self.restore_best_weights:
                self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
        return self.early_stop


class Trainer:
    """A class to handle the training of Variational Autoencoders.

    This class manages the training process, including model training, evaluation,
    logging, and checkpointing. It supports both standard VAE and conditional VAE training.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: Dataset,
        test_data: Dataset,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
        log_dir: Union[str, Path] = "./results",
        run_name: Optional[str] = None,
        num_generated_images: int = 10,
        early_stopping_patience: int = 20,
        is_conditional: bool = False,
        load_checkpoint: Optional[str] = None,
        beta: Optional[float] = 0.1,
        lr_patience: int = 3,
        lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        num_classes: Optional[int] = None,
    ):
        """Initialize the Trainer.

        Args:
            model: The VAE model to train
            train_data: Training dataset
            test_data: Test dataset
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            device: Device to train on
            log_dir: Directory to save logs and checkpoints
            run_name: Optional name for this training run
            num_generated_images: Number of images to generate for visualization
            early_stopping_patience: Number of epochs to wait before early stopping
            is_conditional: Whether the model is a Conditional VAE
            load_checkpoint: Path to checkpoint to load
            beta: KL divergence weight
            lr_patience: Number of epochs to wait before reducing learning rate
            lr_factor: Factor to reduce learning rate by
            min_lr: Minimum learning rate
            num_classes: Number of classes (for Conditional VAE)
        """
        self.model = model.to(device)
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.num_generated_images = num_generated_images
        self.is_conditional = is_conditional
        self.beta = beta
        self.min_lr = min_lr
        self.num_classes = num_classes

        # Calculate model size and parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        # Setup logging directory with timestamp and optional run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{run_name}" if run_name else timestamp
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data loaders
        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )

        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Initialize early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float("inf")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
            "best_model_path": None,
        }
        # Load checkpoint if provided
        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)

        # Log model information
        logger.info(f"Initialized Trainer with model: {model.__class__.__name__}")
        logger.info(f"Training on device: {device}")
        logger.info(f"Logging to: {self.log_dir}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        if is_conditional:
            logger.info("Training Conditional VAE")
            logger.info(f"Number of classes: {num_classes}")

    def train(self, epochs: int) -> Dict[str, Any]:
        """Train the model for the specified number of epochs.

        Args:
            epochs: Number of epochs to train for

        Returns:
            Dictionary containing training history and best model path
        """
        try:
            for _ in range(epochs):
                self.epoch += 1
                logger.info(f"Starting epoch {self.epoch}/{epochs}")

                # Train for one epoch
                train_loss = self._train_epoch()
                self.history["train_loss"].append(train_loss)

                # Evaluate and log
                val_loss = self._eval_and_log()
                self.history["val_loss"].append(val_loss)

                # Step the scheduler
                self.scheduler.step(val_loss)

                # Log the current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/learning_rate", current_lr, self.epoch)
                # Check for early stopping
                if self.early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping triggered after {self.epoch} epochs")
                    break

                # Save checkpoint if this is the best model so far
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.history["best_epoch"] = self.epoch
                    self.history["best_model_path"] = self._save_checkpoint(
                        is_best=True
                    )

                # Save regular checkpoint
                self._save_checkpoint(is_best=False)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise

        # Close tensorboard writer
        self.writer.close()
        return self.history

    def _train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm.tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch in progress_bar:
            loss = self._train_step(batch)
            total_loss += loss
            progress_bar.set_postfix({"loss": loss})

        return total_loss / num_batches

    def _calc_loss(
        self, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the loss for a batch of data.

        Returns:
            Tuple of (KL divergence loss, reconstruction loss, classifier loss)
        """
        if self.is_conditional:
            inputs, conditions = batch
            # Ensure encode() returns exactly 2 values
            mu, log_sigma = self.model.encode(inputs, conditions)
            z = self.model.bottleneck(mu, log_sigma)
            outputs = self.model.decode(z, conditions)

            # Reconstruction loss
            recon_loss = F.mse_loss(outputs, inputs, reduction="sum") / inputs.numel()

            # KL divergence
            kl_loss = (
                -0.5
                * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
                / inputs.numel()
            )

            # Classification loss
            class_logits = self.model.classify_from_latent(z)
            classifier_loss = F.cross_entropy(class_logits, conditions)

            return kl_loss, recon_loss, classifier_loss
        else:
            inputs, _ = batch
            mu, log_sigma = self.model.encode(inputs)
            z = self.model.bottleneck(mu, log_sigma)
            outputs = self.model.decode(z)

            recon_loss = F.mse_loss(outputs, inputs, reduction="sum") / inputs.numel()
            kl_loss = (
                -0.5
                * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
                / inputs.numel()
            )

            return kl_loss, recon_loss, torch.tensor(0.0)  # Dummy classifier loss

    def _train_step(
        self, batch: Union[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """Perform a single training step."""
        self.optimizer.zero_grad()

        if self.is_conditional:
            kl_div_loss, recon_loss, classifier_loss = self._calc_loss(batch)
            loss = recon_loss + self.beta * kl_div_loss + 0.1 * classifier_loss
        else:
            kl_div_loss, recon_loss, _ = self._calc_loss(batch)
            loss = recon_loss + self.beta * kl_div_loss

        loss.backward()
        self.optimizer.step()

        # Log step metrics
        self.writer.add_scalar("train/recon_loss", recon_loss.item(), self.step)
        self.writer.add_scalar("train/kl_div_loss", kl_div_loss.item(), self.step)
        if self.is_conditional:
            self.writer.add_scalar(
                "train/classifier_loss", classifier_loss.item(), self.step
            )
        self.writer.add_scalar("train/loss", loss.item(), self.step)
        self.step += 1

        return loss.item()

    @torch.no_grad()
    def _eval_and_log(self) -> float:
        """Evaluate model on test set and log results."""
        self.model.eval()
        total_loss = 0.0
        val_metrics = []
        total_classifier_loss = 0.0 if self.is_conditional else None

        with torch.no_grad():
            for batch in self.test_loader:
                if self.is_conditional:
                    kl_div_loss, recon_loss, classifier_loss = self._calc_loss(batch)
                    total_loss += (
                        recon_loss + self.beta * kl_div_loss + 0.1 * classifier_loss
                    ).item()
                    total_classifier_loss += classifier_loss.item()
                else:
                    kl_div_loss, recon_loss, _ = self._calc_loss(batch)
                    total_loss += (recon_loss + self.beta * kl_div_loss).item()

                # Calculate per-batch metrics
                if self.is_conditional:
                    inputs, labels = batch
                    outputs, mu, log_sigma = self.model(inputs, labels)
                else:
                    inputs, _ = batch
                    outputs, mu, log_sigma = self.model(inputs)

                batch_metrics = self._calc_metrics(inputs, outputs, mu, log_sigma)
                val_metrics.append(batch_metrics)

        eval_loss = total_loss / len(self.test_loader)

        # Log epoch metrics
        self.writer.add_scalar("val/loss", eval_loss, self.epoch)
        if self.is_conditional and total_classifier_loss is not None:
            avg_classifier_loss = total_classifier_loss / len(self.test_loader)
            self.writer.add_scalar(
                "val/classifier_loss", avg_classifier_loss, self.epoch
            )

        logger.info(
            f"Epoch {self.epoch} - Val Loss: {eval_loss:.4f}"
            + (
                f" | Classifier Loss: {avg_classifier_loss:.4f}"
                if self.is_conditional
                else ""
            )
        )

        # Generate and log sample images
        self._log_generated_images()

        # Log per-batch metrics
        avg_val_metrics = {
            k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0].keys()
        }
        self._log_metrics(avg_val_metrics, self.epoch, "val")

        return eval_loss

    @staticmethod
    def _kl_divergence(log_sigma: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Calculate KL divergence loss.

        Args:
            log_sigma: Log standard deviation of the latent distribution
            mu: Mean of the latent distribution

        Returns:
            KL divergence loss
        """
        return 0.5 * torch.sum((2 * log_sigma).exp() + mu**2 - 1 - 2 * log_sigma)

    @torch.no_grad()
    def _log_generated_images(self):
        """Generate and log sample images to TensorBoard."""
        if self.is_conditional:
            # For conditional VAE, generate samples for each class
            generated_samples = self.generate_class_samples(self.num_generated_images)

            # Log samples for each class
            for class_idx, samples in generated_samples.items():
                for i, img in enumerate(samples):
                    self.writer.add_image(
                        f"generated/class_{class_idx}/{i}", img, global_step=self.epoch
                    )
        else:
            # For standard VAE, use original behavior
            samples = torch.stack(
                [self.test_data[i][0] for i in range(self.num_generated_images)]
            )
            generated = self.model(samples)
            comparison = torch.cat([samples, generated[0]], dim=3)

            for i, img in enumerate(comparison):
                self.writer.add_image(f"generated/{i}", img, global_step=self.epoch)

    def _save_checkpoint(self, is_best: bool = False) -> str:
        """Save a model checkpoint.

        Args:
            is_best: Whether this is the best model so far

        Returns:
            Path to the saved checkpoint
        """
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "history": self.history,
            "is_conditional": self.is_conditional,
        }

        # Save regular checkpoint
        checkpoint_path = self.log_dir / f"checkpoint_epoch_{self.epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model if this is the best so far
        if is_best:
            best_model_path = self.log_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            return str(best_model_path)

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load a model checkpoint for fine-tuning.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint.get(
            "history",
            {
                "train_loss": [],
                "val_loss": [],
                "best_epoch": 0,
                "best_model_path": None,
            },
        )
        self.is_conditional = checkpoint.get("is_conditional", False)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming training from epoch {self.epoch + 1}")
        if self.is_conditional:
            logger.info("Loaded Conditional VAE model")

    def generate_samples(self, num_samples: int = 10) -> torch.Tensor:
        """Generate samples from the trained model.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Tensor containing generated samples
        """
        self.model.eval()
        with torch.no_grad():
            if self.is_conditional:
                # For conditional VAE, generate samples with random conditions
                conditions = torch.randint(0, 10, (num_samples,)).to(self.device)
                samples = self.model.sample(num_samples, conditions)
            else:
                samples = self.model.sample(num_samples)
        return samples

    def _calc_metrics(self, inputs, outputs, mu, log_sigma):
        """Calculate VAE metrics."""
        # Reconstruction metrics
        mse = F.mse_loss(outputs, inputs, reduction="mean")
        psnr = 10 * torch.log10(1.0 / mse)  # Peak Signal-to-Noise Ratio

        # KL divergence loss
        kl_loss = self._kl_divergence(log_sigma, mu) / inputs.numel()

        return {
            "mse": mse.item(),
            "psnr": psnr.item(),
            "kl_loss": kl_loss.item(),
        }

    def _log_metrics(self, metrics, epoch, phase="train"):
        """Log metrics to tensorboard."""
        prefix = f"{phase}/"
        self.writer.add_scalar(f"{prefix}recon_loss", metrics["mse"], epoch)
        self.writer.add_scalar(f"{prefix}psnr", metrics["psnr"], epoch)
        self.writer.add_scalar(f"{prefix}kl_div_loss", metrics["kl_loss"], epoch)

    def _calc_class_metrics(
        self,
        outputs: torch.Tensor,
        inputs: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """Calculate class-specific metrics for Conditional VAE.

        Args:
            outputs: Model outputs
            inputs: Input images
            mu: Mean of latent distribution
            log_sigma: Log standard deviation of latent distribution

        Returns:
            Dictionary of class-specific metrics
        """
        class_metrics = {}
        for i in range(self.num_classes):
            # Get indices for current class
            class_indices = self.current_labels == i

            if class_indices.any():
                # Calculate class-specific MSE
                class_mse = F.mse_loss(
                    outputs[class_indices], inputs[class_indices], reduction="mean"
                ).item()

                # Calculate class-specific PSNR
                class_psnr = (
                    10 * torch.log10(1.0 / class_mse) if class_mse > 0 else float("inf")
                )

                # Calculate class-specific KL divergence
                class_kl = self._kl_divergence(
                    log_sigma[class_indices], mu[class_indices]
                ).item()

                # Store class-specific metrics
                class_metrics.update(
                    {
                        f"class_{i}/mse": class_mse,
                        f"class_{i}/psnr": class_psnr,
                        f"class_{i}/kl_loss": class_kl,
                    }
                )

        return class_metrics

    def _log_class_metrics(self, metrics, epoch, phase="train"):
        """Log class-specific metrics to tensorboard."""
        prefix = f"{phase}/class_"
        for i in range(self.num_classes):
            for name, value in metrics.items():
                if name.startswith(f"{prefix}{i}/"):
                    self.writer.add_scalar(name, value, epoch)

    def generate_class_samples(self, num_samples: int = 10) -> Dict[int, torch.Tensor]:
        """Generate samples for each class using the trained CVAE.

        Args:
            num_samples: Number of samples to generate per class

        Returns:
            Dictionary mapping class indices to generated samples
        """
        if not self.is_conditional:
            raise ValueError("This method is only available for Conditional VAE")

        self.model.eval()
        generated_samples = {}

        with torch.no_grad():
            for class_idx in range(self.num_classes):
                # Create condition tensor for this class
                conditions = torch.full(
                    (num_samples,), class_idx, device=self.device, dtype=torch.long
                )

                # Generate samples for this class
                samples = self.model.generate(conditions, num_samples)
                generated_samples[class_idx] = samples

        return generated_samples

    def save_generated_samples(
        self, output_dir: Union[str, Path], num_samples: int = 10
    ):
        """Save generated samples to disk for visualization.

        Args:
            output_dir: Directory to save generated samples
            num_samples: Number of samples to generate per class
        """
        if not self.is_conditional:
            raise ValueError("This method is only available for Conditional VAE")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate samples for each class
        generated_samples = self.generate_class_samples(num_samples)

        # Save samples for each class
        for class_idx, samples in generated_samples.items():
            class_dir = output_dir / f"class_{class_idx}"
            class_dir.mkdir(exist_ok=True)

            for i, img in enumerate(samples):
                # Convert tensor to PIL Image
                img = img.cpu().numpy()
                img = ((img + 1) * 127.5).astype(
                    np.uint8
                )  # Scale from [-1, 1] to [0, 255]
                img = img.transpose(1, 2, 0)  # Change from (C, H, W) to (H, W, C)

                # Save image
                Image.fromarray(img).save(class_dir / f"sample_{i}.png")

        logger.info(f"Saved generated samples to {output_dir}")
