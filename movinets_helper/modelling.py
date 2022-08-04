"""Call the models in a single function ready to be trained.

These are just a bunch of wrapper functions on the Movinet model.
"""

from dataclasses import dataclass

import tensorflow as tf

from official.vision.configs import video_classification
from official.projects.movinet.configs import movinet as movinet_configs
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_layers
from official.projects.movinet.modeling import movinet_model


@dataclass
class BaseConfig:
    """Configuration for the data ingested to the model. """
    num_frames: int = 10
    resolution: int = 172
    batch_size: int = 8
    channels: int = 3


@dataclass
class ConfigMovinetA0Base(BaseConfig):
    id: int = 0


@dataclass
class ConfigMovinetA2Base(BaseConfig):
    id: int = 1


@dataclass
class ConfigMovinetA2Base(BaseConfig):
    id: int = 2
    resolution: int = 224


def build_classifier(
    backbone: movinet.Movinet,
    num_classes: int,
    model_config: BaseConfig,
    freeze_backbone: int = True,
    stream: bool = False,
):
    """Builds a classifier on top of a backbone model.

    Copied from movinet_tutorial.ipynb.
    Wraps the backbone with a new classifier to crete a new classifier
    head with num_classes outputs. Freezes all layers but the last.

    Args:
        backbone (movinet.Movinet): _description_
        num_classes (int): Number of classes for your given model.
        model_config (BaseConfig): Configuration of the model chosen.
            Contains 
        freeze_backbone (int, optional):
            Whether to freeze all the layers but the last. Defaults to True.
        stream (bool, optional):
            Use base or stream model. Only implemented base, the parameter
            is here for future use.
            Defaults to False.

    Returns:
        _type_: Model ready to be compiled.
    """

    model = movinet_model.MovinetClassifier(
        backbone=backbone, num_classes=num_classes, output_states=stream
    )
    model.build(
        [
            model_config.batch_size,
            model_config.num_frames,
            model_config.resolution,
            model_config.resolution,
            model_config.channels,
        ]
    )

    # Freeze all layers but the last for fine-tuning
    if freeze_backbone:
        for layer in model.layers[:-1]:
            layer.trainable = False

    model.layers[-1].trainable = True

    return model


def make_model(
    num_classes: int, model_id: str = "a0", version: str = "base", freeze_backbone: bool = True
):
    if version != "base":
        raise ValueError(f"Movinet Stream model not implemented")

    backbone = movinet.Movinet(model_id=model_id)
    # Initial number of classes 600 from Kinetics
    model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
    model.build([1, 1, 1, 1, 3])

    # TODO: REVIEW THE CHECKPOINT
    checkpoint_dir = f"movinet_{model_id}_{version}"
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

    checkpoint = tf.train.Checkpoint(model=model)

    status = checkpoint.restore(checkpoint_path)

    model = build_classifier(backbone, num_classes, freeze_backbone=True)
    return model


def default_hyperparams():
    pass


def default_loss_function():
    pass


def default_metrics():
    pass


def default_learning_rate():
    pass


def default_optimizer():
    pass


def default_callbacks():
    pass
