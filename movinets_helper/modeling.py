"""Call the models in a single function ready to be trained.

These are just a bunch of wrapper functions on the Movinet model.
"""

from typing import *
from dataclasses import dataclass

import tensorflow as tf

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model


@dataclass
class BaseConfig:
    """Configuration for the data ingested to the model."""
    num_frames: int = 10
    resolution: int = 172
    batch_size: int = 8
    channels: int = 3
    epochs: int = 3
    version: str = "base"


@dataclass
class ConfigMovinetA0Base(BaseConfig):
    model_id: int = "a0"


@dataclass
class ConfigMovinetA2Base(BaseConfig):
    model_id: int = "a0"


@dataclass
class ConfigMovinetA2Base(BaseConfig):
    model_id: int = "a2"
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
    num_classes: int,
    config: BaseConfig,
    freeze_backbone: bool = True,
):
    if config.version != "base":
        raise ValueError(f"Movinet Stream model not implemented")

    backbone = movinet.Movinet(model_id=config.model_id)
    # Initial number of classes 600 from Kinetics
    model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
    model.build([1, 1, 1, 1, 3])

    # TODO: REVIEW THE CHECKPOINT
    checkpoint_dir = f"movinet_{config.model_id}_{config.version}"
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

    checkpoint = tf.train.Checkpoint(model=model)

    status = checkpoint.restore(checkpoint_path)

    model = build_classifier(backbone, num_classes, config, freeze_backbone=freeze_backbone)
    return model


def default_hyperparams(total_train_steps: int):
    """Gathers the hyperparameters used in the movinet tutorial to train
    the models. 

    Just call this if you aren't willing to investigate further.

    FIXME:
        - Explain how to use it
        - Set examples
    """
    return {
        "loss_function": loss_function(),
        "metrics": metrics(),
        "optimizer": optimizer(total_train_steps),
        "callbacks": loss_function(),
    }


def loss_function() -> tf.keras.losses.CategoricalCrossentropy:
    return tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1
    )


def metrics() -> List[tf.keras.metrics.TopKCategoricalAccuracy]:
    return [
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top_1", dtype=tf.float32),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5", dtype=tf.float32),
    ]


def learning_rate(total_train_steps: int):
    initial_learning_rate = 0.01
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate,
        decay_steps=total_train_steps,
    )


def optimizer(total_train_steps: int):
    lr = learning_rate(total_train_steps)
    return tf.keras.optimizers.RMSprop(
        lr, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0
    )


def callbacks(checkpoint_filepath: str) -> List[tf.keras.callbacks.Callback]:
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_top_1',
        mode='max',
        save_best_only=True
    )

    return [
        tf.keras.callbacks.TensorBoard(),
        model_checkpoint_callback
    ]
