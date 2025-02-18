"""Random Forest implementation for geospatial segmentation."""
import numpy as np
from typing import Dict, Optional, List, Tuple
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .base import GeospatialModel
from ..utils.logger import setup_logger
from .. import config

logger = setup_logger(__name__)

class RandomForestModel(GeospatialModel):
    """Random Forest classifier for geospatial segmentation."""

    def __init__(
        self,
        n_estimators: int = config.RF_N_ESTIMATORS,
        max_depth: int = config.RF_MAX_DEPTH,
        max_features: str = 'sqrt',
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Optional[str] = 'balanced'
    ):
        """Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            max_features: Number of features to consider for best split
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in leaf node
            class_weight: Sample weighting mode
        """
        super().__init__()

        self.parameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'max_features': max_features,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight
        }

        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1  # Use all available cores
        )

        logger.debug(f"Initialized RandomForestModel with {n_estimators} trees")

    def train(
        self,
        image: np.ndarray,
        labels: Dict[str, np.ndarray],
        mask: Optional[np.ndarray] = None,
        validation_split: float = 0.2,
        **kwargs
    ) -> Dict:
        """Train the Random Forest classifier.

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
            logger.info("Starting Random Forest training")
            self.validate_image(image)

            # Prepare training data
            X, y = self.prepare_training_data(image, labels, mask)

            # Split into training and validation sets
            if validation_split > 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=validation_split,
                    stratify=y,
                    random_state=42
                )
            else:
                X_train, y_train = X, y
                X_val, y_val = None, None

            # Update model parameters from kwargs
            if kwargs:
                self.classifier.set_params(**kwargs)
                self.parameters.update(kwargs)

            # Train classifier
            logger.debug("Fitting Random Forest classifier")
            self.classifier.fit(X_train, y_train)

            # Calculate feature importances
            importances = self.classifier.feature_importances_
            feature_names = []
            for extractor_name, features in self.get_feature_info().items():
                feature_names.extend([f"{extractor_name}_{f}" for f in features])

            importance_dict = dict(zip(feature_names, importances))

            # Log top 10 most important features
            top_features = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            logger.debug("Top 10 most important features:")
            for name, importance in top_features:
                logger.debug(f"{name}: {importance:.4f}")

            # Calculate training metrics
            metrics = {
                'feature_importances': importance_dict,
                'training_score': self.classifier.score(X_train, y_train)
            }

            # Calculate validation metrics if available
            if X_val is not None and y_val is not None:
                val_score = self.classifier.score(X_val, y_val)
                val_pred = self.classifier.predict(X_val)

                metrics['validation_score'] = val_score
                metrics['classification_report'] = classification_report(
                    y_val,
                    val_pred,
                    target_names=self.class_names,
                    output_dict=True
                )

                logger.debug(f"Validation score: {val_score:.4f}")

            self.is_trained = True
            self.model_info = {
                'parameters': self.parameters,
                'n_features': X.shape[1],
                'n_classes': len(self.class_names),
                'feature_importances': importance_dict,
                'metrics': metrics
            }

            logger.info("Random Forest training completed")
            return metrics

        except Exception as e:
            logger.error("Error during Random Forest training")
            logger.exception(e)
            raise

    def predict(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict segmentation using Random Forest.

        Args:
            image: Image to segment
            mask: Optional mask of valid data areas

        Returns:
            numpy.ndarray: Predicted segmentation mask
        """
        try:
            logger.debug("Starting Random Forest prediction")
            self.validate_image(image)

            if not self.is_trained:
                raise RuntimeError("Model must be trained before prediction")

            # Extract features
            feature_dict = self.feature_set.extract_all_features(image)
            features = np.vstack([f.reshape(f.shape[0], -1)
                                for f in feature_dict.values()])

            # Reshape to (n_pixels, n_features)
            n_pixels = image.shape[0] * image.shape[1]
            features = features.reshape(features.shape[0], n_pixels).T

            # Apply mask if provided
            if mask is not None:
                valid_pixels = mask.ravel()
                features = features[valid_pixels]

            # Predict
            logger.debug("Running prediction")
            if mask is not None:
                # Initialize output array
                pred = np.zeros(n_pixels, dtype=int)
                # Predict only for valid pixels
                pred[valid_pixels] = self.classifier.predict(features)
            else:
                pred = self.classifier.predict(features)

            # Reshape to original dimensions
            pred = pred.reshape(image.shape[:2])

            logger.debug("Prediction completed")
            return pred

        except Exception as e:
            logger.error("Error during Random Forest prediction")
            logger.exception(e)
            raise

    def save_model(self, path: str) -> None:
       """Save Random Forest model to disk.

       Args:
           path: Save path
       """
       try:
           logger.info(f"Saving model to {path}")

           save_dict = {
               'classifier': self.classifier,
               'parameters': self.parameters,
               'class_names': self.class_names,
               'model_info': self.model_info,
               'is_trained': self.is_trained,
               'feature_extractors': {
                   name: extractor.get_parameters()
                   for name, extractor in self.feature_set.extractors.items()
               }
           }

           joblib.dump(save_dict, path)
           logger.info("Model saved successfully")

       except Exception as e:
           logger.error(f"Error saving model to {path}")
           logger.exception(e)
           raise

    def load_model(self, path: str) -> None:
       """Load Random Forest model from disk.

       Args:
           path: Load path
       """
       try:
           logger.info(f"Loading model from {path}")

           load_dict = joblib.load(path)

           self.classifier = load_dict['classifier']
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
            features = np.vstack([f.reshape(f.shape[0], -1)

            for f in feature_dict.values()])

            # Reshape to (n_pixels, n_features)
            n_pixels = image.shape[0] * image.shape[1]
            features = features.reshape(features.shape[0], n_pixels).T

            # Apply mask if provided
            if mask is not None:
                valid_pixels = mask.ravel()
                features = features[valid_pixels]

            # Predict
            logger.debug("Running prediction")
            if mask is not None:
                # Initialize output array
                pred = np.zeros(n_pixels, dtype=int)
                # Predict only for valid pixels
                pred[valid_pixels] = self.classifier.predict(features)
            else:
                pred = self.classifier.predict(features)

            # Reshape to original dimensions
            pred = pred.reshape(image.shape[:2])

            logger.debug("Prediction completed")
            return pred

        except Exception as e:
            logger.error("Error during Random Forest prediction")
            logger.exception(e)
            raise
