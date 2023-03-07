"""Stardist 3D model with new weights and probability maps."""
from unittest.mock import patch

from stardist.models import StarDist3D  # type: ignore [import]

from .data_3d import OptimizedStarDistData3D


class OptimizedStarDist3D(StarDist3D):  # type: ignore [misc]
    """Overwrite train method to use different data generator."""

    def train(  # type: ignore [no-untyped-def]
        self,
        x,
        y,
        validation_data,
        classes="auto",
        augmenter=None,
        seed=None,
        epochs=None,
        steps_per_epoch=None,
        workers=1,
    ):
        """Monkey patch the original StarDistData3D generator."""
        with patch("stardist.models.model3d.StarDistData3D", OptimizedStarDistData3D):
            return super().train(
                X=x,
                Y=y,
                validation_data=validation_data,
                classes=classes,
                augmenter=augmenter,
                seed=seed,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                workers=workers,
            )
