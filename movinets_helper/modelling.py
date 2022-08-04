"""Call the models in a single function ready to be trained.

These are just a bunch of wrapper functions on the Movinet model.
"""

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class BaseConfig:
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
    resolution: int = 224
    id: int = 2
