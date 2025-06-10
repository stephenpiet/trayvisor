import os
from typing import Tuple, Optional, Dict, Any, Union, List
import logging
from pathlib import Path
from datetime import datetime
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to monitor training and stop when validation loss stops improving."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
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
        self.best_loss = float('inf')
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
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        device: torch.device,
        log_dir: Union[str, Path] = './results',
        run_name: Optional[str] = None,
        num_generated_images: int = 10,
        early_stopping_patience: int = 5,
        is_conditional: bool = False,
        load_checkpoint: Optional[str] = None
    ):
        """Initialize the Trainer.
        
        Args:
            model: The VAE model to train
            train_data: Training dataset
            test_data: Test dataset
            optimizer: Optimizer for training
            batch_size: Batch size for training
            device: Device to train on
            log_dir: Base directory for logging and checkpoints
            run_name: Optional name for this training run
            num_generated_images: Number of images to generate for visualization
            early_stopping_patience: Number of epochs to wait before early stopping
            is_conditional: Whether the model is a Conditional VAE
            load_checkpoint: Path to checkpoint to load for fine-tuning
        """
        self.model = model.to(device)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.num_generated_images = num_generated_images
        self.is_conditional = is_conditional
        
        # Setup logging directory with timestamp and optional run name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{timestamp}_{run_name}" if run_name else timestamp
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loaders
        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0,
            'best_model_path': None
        }
        
        # Load checkpoint if provided
        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)
        
        logger.info(f"Initialized Trainer with model: {model.__class__.__name__}")
        logger.info(f"Training on device: {device}")
        logger.info(f"Logging to: {self.log_dir}")
        if is_conditional:
            logger.info("Training Conditional VAE")

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
                self.history['train_loss'].append(train_loss)
                
                # Evaluate and log
                val_loss = self._eval_and_log()
                self.history['val_loss'].append(val_loss)
                
                # Check for early stopping
                if self.early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping triggered after {self.epoch} epochs")
                    break
                
                # Save checkpoint if this is the best model so far
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.history['best_epoch'] = self.epoch
                    self.history['best_model_path'] = self._save_checkpoint(is_best=True)
                
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
        
        progress_bar = tqdm.tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch in progress_bar:
            loss = self._train_step(batch)
            total_loss += loss
            progress_bar.set_postfix({'loss': loss})
            
        return total_loss / num_batches

    def _train_step(self, batch: Union[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Perform a single training step.
        
        Args:
            batch: Tuple of (inputs, targets) tensors for standard VAE
                  or (inputs, conditions) tensors for Conditional VAE
            
        Returns:
            Loss value for the step
        """
        self.optimizer.zero_grad()
        
        kl_div_loss, recon_loss = self._calc_loss(batch)
        loss = recon_loss + kl_div_loss
        loss.backward()
        self.optimizer.step()

        # Log step metrics
        self.writer.add_scalar('train/recon_loss', recon_loss.item(), self.step)
        self.writer.add_scalar('train/kl_div_loss', kl_div_loss.item(), self.step)
        self.writer.add_scalar('train/loss', loss.item(), self.step)
        self.step += 1
        
        return loss.item()

    def _calc_loss(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for a batch of data.
        
        Args:
            batch: Input data batch
            
        Returns:
            Tuple of (KL divergence loss, reconstruction loss)
        """
        inputs, _ = batch
        mu, log_sigma = self.model.encode(inputs)
        latent_code = self.model.bottleneck(mu, log_sigma)
        outputs = self.model.decode(latent_code)
        
        recon_loss = F.mse_loss(outputs, inputs, reduction='sum')
        kl_div_loss = self._kl_divergence(log_sigma, mu)

        return kl_div_loss, recon_loss

    @staticmethod
    def _kl_divergence(log_sigma: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Calculate KL divergence loss.
        
        Args:
            log_sigma: Log standard deviation of the latent distribution
            mu: Mean of the latent distribution
            
        Returns:
            KL divergence loss
        """
        return 0.5 * torch.sum((2 * log_sigma).exp() + mu ** 2 - 1 - 2 * log_sigma)

    @torch.no_grad()
    def _eval_and_log(self) -> float:
        """Evaluate model on test set and log results.
        
        Returns:
            Average evaluation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        # Calculate evaluation loss
        for batch in self.test_loader:
            kl_div_loss, recon_loss = self._calc_loss(batch)
            total_loss += (kl_div_loss + recon_loss).item()
        
        eval_loss = total_loss / len(self.test_loader)
        
        # Log epoch metrics
        self.writer.add_scalar('val/loss', eval_loss, self.epoch)
        logger.info(f"Epoch {self.epoch} - Val Loss: {eval_loss:.4f}")
        
        # Generate and log sample images
        self._log_generated_images()
        
        return eval_loss

    @torch.no_grad()
    def _log_generated_images(self):
        """Generate and log sample images to TensorBoard."""
        samples = torch.stack([self.test_data[i][0] for i in range(self.num_generated_images)])
        generated = self.model(samples)
        comparison = torch.cat([samples, generated], dim=3)
        
        for i, img in enumerate(comparison):
            self.writer.add_image(f'generated/{i}', img, global_step=self.epoch)

    def _save_checkpoint(self, is_best: bool = False) -> str:
        """Save a model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'is_conditional': self.is_conditional
        }
        
        # Save regular checkpoint
        checkpoint_path = self.log_dir / f'checkpoint_epoch_{self.epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_model_path = self.log_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            return str(best_model_path)
        
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load a model checkpoint for fine-tuning.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint.get('history', {
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0,
            'best_model_path': None
        })
        self.is_conditional = checkpoint.get('is_conditional', False)
        
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