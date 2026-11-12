"""Layer: base class for map layers (name, visibility, opacity, CRS, render cache)."""
from abc import abstractmethod
from typing import Optional, Tuple
from PyQt6.QtGui import QPainter
from PyQt6.QtCore import QObject, pyqtSignal
from ..utils.logger import setup_logger
from .. import config

logger = setup_logger(__name__)


class Layer(QObject):
    """Base class for all map layers."""

    # Signals
    changed = pyqtSignal()  # Emitted when layer changes
    visibility_changed = pyqtSignal(bool)  # Emitted when visibility changes

    def __init__(self, name: str):
        """Initialize layer.

        Args:
            name: Layer name
        """
        super().__init__()
        self._name = name
        self._visible = True
        self._opacity = 1.0
        self._crs = config.DEFAULT_CRS

        # Caching for rendering optimization
        self._cache = {}
        self._cache_valid = False


    def __init_subclass__(cls, **kwargs):
        """Track subclass creation to detect potential overrides."""
        super().__init_subclass__(**kwargs)
        logger.debug(f"Layer subclass created: {cls.__name__}")

        # Check if visible property is overridden
        if 'visible' in cls.__dict__:
            logger.debug(f"  WARNING: {cls.__name__} overrides the 'visible' property")

        # Check if _visible attribute is accessed differently
        for method_name, method in cls.__dict__.items():
            if callable(method) and method_name not in ('__init__', '__init_subclass__'):
                logger.debug(f"  Method {cls.__name__}.{method_name} defined")

    @property
    def name(self) -> str:
        """Get layer name."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set layer name."""
        self._name = value
        self.changed.emit()

    @property
    def visible(self) -> bool:
        """Get layer visibility."""
        return self._visible



    @visible.setter
    def visible(self, value: bool):
        """Set layer visibility."""
        logger.debug(f"\n==== VISIBILITY SETTER called for {self.name} ====")
        logger.debug(f"Setting visibility from {self._visible} to {value}")
        logger.debug(f"Layer class: {self.__class__.__name__}")

        # Check if there's an actual change
        if self._visible != value:
            logger.debug(f"Changing visibility value")
            self._visible = value

            # Verify the change took effect
            logger.debug(f"_visible after change: {self._visible}")
            logger.debug(f"visible property returns: {self.visible}")

            # Emit signals (with verification)
            logger.debug(f"Emitting visibility_changed signal with value={value}")
            self.visibility_changed.emit(value)

            logger.debug(f"Emitting changed signal")
            self.changed.emit()

            # Invalidate cache
            old_cache_valid = self._cache_valid
            self._cache_valid = False
            logger.debug(f"Cache invalidated: {old_cache_valid} -> {self._cache_valid}")

            logger.debug(f"Layer {self.name} visibility changed to {value}")
        else:
            logger.debug(f"No change needed, visibility already {value}")

        logger.debug(f"==== VISIBILITY SETTER completed ====\n")

    @property
    def opacity(self) -> float:
        """Get layer opacity."""
        return self._opacity

    @opacity.setter
    def opacity(self, value: float):
        """Set layer opacity."""
        self._opacity = max(0.0, min(1.0, value))
        self._cache_valid = False
        self.changed.emit()

    @property
    def crs(self) -> str:
        """Get layer CRS."""
        return self._crs

    @crs.setter
    def crs(self, value: str):
        """Set layer CRS."""
        self._crs = value
        self._cache_valid = False
        self.changed.emit()

    @abstractmethod
    def extent(self) -> Optional[Tuple[float, float, float, float]]:
        """Get layer extent (min_lon, min_lat, max_lon, max_lat)."""
        pass

    @abstractmethod
    def render(self, painter: QPainter, map_canvas) -> None:
        """Render layer to map canvas.

        Args:
            painter: QPainter object
            map_canvas: MapCanvas widget
        """
        pass

    def invalidate_cache(self):
        """Invalidate render cache."""
        self._cache_valid = False
        self._cache.clear()
