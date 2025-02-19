"""CNN implementation for geospatial segmentation."""
import numpy as np
from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from .base import GeospatialModel
from ..utils.logger import setup_logger
from .. import config

logger = setup_logger(__name__)

class UNet(nn.Module):
    """U-Net architecture for semantic segmentation."""

    def __init__(self, n_channels: int, n_classes: int):
        """Initialize U-Net.

        Args:
            n_channels: Number of input channels
            n_classes: Number of output classes
        """
        super().__init__()

        # Encoder path
        self.enc1 = self._make_layer(n_channels, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)

        # Bridge
        self.bridge = self._make_layer(512, 1024)

        # Decoder path
        self.dec4 = self._make_layer(1024 + 512, 512)
        self.dec3 = self._make_layer(512 + 256, 256)
        self.dec2 = self._make_layer(256 + 128, 128)
        self.dec1 = self._make_layer(128 + 64, 64)

        # Final layer
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

        # Initialize weights
        self.apply(self._init_weights)

    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels

        Returns:
            nn.Sequential: Convolutional block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self, m):
        """Initialize layer weights.

        Args:
            m: Layer
        """
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bridge
        bridge = self.bridge(F.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.dec4(torch.cat([F.interpolate(bridge, size=enc4.shape[2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, size=enc3.shape[2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, size=enc2.shape[2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, size=enc1.shape[2:]), enc1], 1))

        return self.final(dec1)

class SegmentationDataset(Dataset):
    """Dataset for training segmentation model."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        patch_size: int = 256,
        stride: int = 128
    ):
        """Initialize dataset.

        Args:
            features: Feature array (channels, height, width)
            labels: Label array (height, width)
            patch_size: Size of image patches
            stride: Stride for patch extraction
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        self.patch_size = patch_size
        self.stride = stride

        # Calculate patch positions
        self.patches = []
        for i in range(0, features.shape[1] - patch_size + 1, stride):
            for j in range(0, features.shape[2] - patch_size + 1, stride):
                self.patches.append((i, j))

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item.

        Args:
            idx: Item index

        Returns:
            tuple: (feature patch, label patch)
        """
        i, j = self.patches[idx]
        feature_patch = self.features[:, i:i+self.patch_size, j:j+self.patch_size]
        label_patch = self.labels[i:i+self.patch_size, j:j+self.patch_size]
        return feature_patch, label_patch

class CNNModel(GeospatialModel):
    """CNN model for geospatial segmentation."""

    def __init__(
        self,
        patch_size: int = 256,
        batch_size: int = config.CNN_BATCH_SIZE,
        learning_rate: float = config.CNN_LEARNING_RATE,
        n_epochs: int = config.CNN_EPOCHS
    ):
        """Initialize CNN model.

        Args:
            patch_size: Size of image patches for training
            batch_size: Training batch size
            learning_rate: Learning rate
            n_epochs: Number of training epochs
        """
        super().__init__()

        self.parameters = {
            'patch_size': patch_size,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'n_epochs': n_epochs
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        logger.debug(f"Initialized CNNModel (device: {self.device})")

    def train(
        self,
        image: np.ndarray,
        labels: Dict[str, np.ndarray],
        mask: Optional[np.ndarray] = None,
        validation_split: float = 0.2,
        **kwargs
    ) -> Dict:
        """Train the CNN model.

        Args:
            image: Training image
            labels: Dictionary mapping class names to boolean masks
            mask: Optional mask of valid data areas
            validation_split: Fraction of data to use for validation
            **kwargs: Additional training parameters

        Returns:
            dict: Training metrics
        """
        try:
            logger.info("Starting CNN training")
            self.validate_image(image)

            # Update parameters from kwargs
            if kwargs:
                self.parameters.update(kwargs)

            # Extract features
            feature_dict = self.feature_set.extract_all_features(image)
            features = np.vstack([f for f in feature_dict.values()])

            # Store class names
            self.class_names = list(labels.keys())
            logger.debug(f"Training with classes: {self.class_names}")

            # Create label array starting from 0 (background)
            label_array = np.zeros(image.shape[:2], dtype=np.int64)
            for i, (name, mask) in enumerate(labels.items()):
                label_array[mask] = i  # Use 0-based indexing for PyTorch


            unique_labels = np.unique(label_array)
            logger.debug(f"Unique label indices in array: {unique_labels}")
            logger.debug(f"Label array shape: {label_array.shape}, dtype: {label_array.dtype}")

            if len(self.class_names) == 0:
                raise ValueError("No classes found in training data")

            # Create datasets
            dataset = SegmentationDataset(
                features,
                label_array,
                patch_size=self.parameters['patch_size']
            )

            # Split into train/val
            train_size = int((1 - validation_split) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.parameters['batch_size'],
                shuffle=True,
                num_workers=4
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.parameters['batch_size'],
                shuffle=False,
                num_workers=4
            )

            # Initialize model if needed
            if self.model is None:
                # Create model with correct number of classes
                self.model = UNet(
                    n_channels=features.shape[0],
                    n_classes=max(2, len(self.class_names))  # At least 2 classes (background + 1)
                ).to(self.device)

                logger.debug(f"Created UNet with {features.shape[0]} input channels and {len(self.class_names)} output classes")

            # Set up training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.parameters['learning_rate']
            )

            # Training loop
            metrics = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

            for epoch in range(self.parameters['n_epochs']):
                # Training
                self.model.train()
                train_loss = 0

                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)
                metrics['train_loss'].append(train_loss)

                # Validation
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)

                        outputs = self.model(batch_features)
                        loss = criterion(outputs, batch_labels)
                        val_loss += loss.item()

                        _, predicted = outputs.max(1)
                        total += batch_labels.numel()
                        correct += predicted.eq(batch_labels).sum().item()

                val_loss /= len(val_loader)
                val_accuracy = 100.0 * correct / total

                metrics['val_loss'].append(val_loss)
                metrics['val_accuracy'].append(val_accuracy)

                logger.debug(
                    f"Epoch {epoch+1}/{self.parameters['n_epochs']} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.2f}%"
                )

            self.is_trained = True
            self.model_info = {
                'parameters': self.parameters,
                'n_features': features.shape[0],
                'n_classes': len(self.class_names) + 1,
                'metrics': metrics,
                'final_accuracy': metrics['val_accuracy'][-1]
            }

            logger.info("CNN training completed")
            return metrics

        except Exception as e:
            logger.error("Error during CNN training")
            logger.exception(e)
            raise

    def predict(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict segmentation using CNN.

        Args:
            image: Image to segment
            mask: Optional mask of valid data areas

        Returns:
            numpy.ndarray: Predicted segmentation mask
        """
        try:
            logger.debug("Starting CNN prediction")
            self.validate_image(image)

            if not self.is_trained:
                raise RuntimeError("Model must be trained before prediction")

            # Extract features
            feature_dict = self.feature_set.extract_all_features(image)
            features = np.vstack([f for f in feature_dict.values()])

            # Convert to tensor
            features = torch.from_numpy(features).float()

            # Create patches
            patch_size = self.parameters['patch_size']
            height, width = image.shape[:2]

            # Add padding if needed
            pad_h = (patch_size - height % patch_size) % patch_size
            pad_w = (patch_size - width % patch_size) % patch_size

            if pad_h > 0 or pad_w > 0:
                features = F.pad(features, (0, pad_w, 0, pad_h), mode='reflect')

            # Predict in patches
            self.model.eval()
            predictions = np.zeros((height + pad_h, width + pad_w), dtype=np.int64)

            with torch.no_grad():
                for i in range(0, features.shape[1], patch_size):
                    for j in range(0, features.shape[2], patch_size):
                        patch = features[:, i:i+patch_size, j:j+patch_size]
                        patch = patch.unsqueeze(0).to(self.device)

                        output = self.model(patch)
                        pred = output.argmax(1).squeeze().cpu().numpy()
                        predictions[i:i+patch_size, j:j+patch_size] = pred

            # Remove padding
            predictions = predictions[:height, :width]

            # Apply mask if provided
            if mask is not None:
                predictions[~mask] = 0

            logger.debug("Prediction completed")
            return predictions

        except Exception as e:
            logger.error("Error during CNN prediction")
            logger.exception(e)
            raise

    def predict_proba(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            image: Image to segment
            mask: Optional mask of valid data areas

        Returns:
            numpy.ndarray: Probability maps for each class
        """
        try:
            logger.debug("Starting probability prediction")
            self.validate_image(image)

            if not self.is_trained:
                raise RuntimeError("Model must be trained before prediction")

            # Extract features
            feature_dict = self.feature_set.extract_all_features(image)
            features = np.vstack([f for f in feature_dict.values()])

            # Convert to tensor
            features = torch.from_numpy(features).float()

            # Create patches
            patch_size = self.parameters['patch_size']
            height, width = image.shape[:2]
            n_classes = len(self.class_names)  # Number of classes without background

            # Add padding if needed
            pad_h = (patch_size - height % patch_size) % patch_size
            pad_w = (patch_size - width % patch_size) % patch_size

            if pad_h > 0 or pad_w > 0:
                features = F.pad(features, (0, pad_w, 0, pad_h), mode='reflect')

            # Predict probabilities in patches
            self.model.eval()
            probabilities = np.zeros((n_classes, height + pad_h, width + pad_w))

            with torch.no_grad():
                for i in range(0, features.shape[1], patch_size):
                    for j in range(0, features.shape[2], patch_size):
                        patch = features[:, i:i+patch_size, j:j+patch_size]
                        patch = patch.unsqueeze(0).to(self.device)

                        output = self.model(patch)
                        proba = F.softmax(output, dim=1).squeeze().cpu().numpy()
                        probabilities[:, i:i+patch_size, j:j+patch_size] = proba

            # Remove padding
            probabilities = probabilities[:, :height, :width]

            # Apply mask if provided
            if mask is not None:
                probabilities[:, ~mask] = 0

            logger.debug("Probability prediction completed")
            return probabilities

        except Exception as e:
            logger.error("Error during probability prediction")
            logger.exception(e)
            raise

    def save_model(self, path: str) -> None:
        """Save CNN model to disk.

        Args:
            path: Save path
        """
        try:
            logger.info(f"Saving model to {path}")

            save_dict = {
                'model_state': self.model.state_dict(),
                'parameters': self.parameters,
                'class_names': self.class_names,
                'model_info': self.model_info,
                'is_trained': self.is_trained,
                'feature_extractors': {
                    name: extractor.get_parameters()
                    for name, extractor in self.feature_set.extractors.items()
                }
            }

            torch.save(save_dict, path)
            logger.info("Model saved successfully")

        except Exception as e:
            logger.error(f"Error saving model to {path}")
            logger.exception(e)
            raise

    def load_model(self, path: str) -> None:
        """Load CNN model from disk.

        Args:
            path: Load path
        """
        try:
            logger.info(f"Loading model from {path}")

            load_dict = torch.load(path, map_location=self.device)

            # Initialize model if needed
            if self.model is None:
                self.model = UNet(
                    n_channels=load_dict['model_info']['n_features'],
                    n_classes=load_dict['model_info']['n_classes']
                ).to(self.device)

            self.model.load_state_dict(load_dict['model_state'])
            self.parameters = load_dict['parameters']
            self.class_names = load_dict['class_names']
            self.model_info = load_dict['model_info']
            self.is_trained = load_dict['is_trained']

            # Restore feature extractor parameters
            for name, params in load_dict['feature_extractors'].items():
                if name in self.feature_set.extractors:
                    self.feature_set.extractors[name].set_parameters(params)

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model from {path}")
            logger.exception(e)
            raise
