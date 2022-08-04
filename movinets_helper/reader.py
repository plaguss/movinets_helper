"""Functionalities to read a TFRecordDataset ready to train a network. """

from typing import *

import tensorflow as tf


def encode_label(label: str, num_classes: int) -> tf.Tensor:
    """One hot encodes the labels according to the number of classes.

    Args:
        label (str): Label representing the movement of the video.
        num_classes (int): Total number of classes in the dataset.

    Returns:
        tf.Tensor: Encoded representation of the label
    """
    return tf.one_hot(label, num_classes)


def _parse_example(example_proto) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decodes the proto files, called by get_dataset.

    During the evaluation time, it's not possible to see the results.
    Only after the dataset is generated and iterated you can get the values.

    Args:
        example_proto (_type_)

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: video and label parsed.
        The video will get the shape internally stored on the video.
        Any unknown shape is informed as None.
    """
    feature_description = {
        "label": tf.io.FixedLenFeature([], tf.int64),
        "n_frames": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "n_channels": tf.io.FixedLenFeature([], tf.int64),
        "video": tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_example(example_proto, feature_description)
    label = tf.cast(parsed["label"], tf.int32)
    n_frames = tf.cast(parsed["n_frames"], tf.int32)
    width = tf.cast(parsed["width"], tf.int32)
    height = tf.cast(parsed["height"], tf.int32)

    video = tf.io.parse_tensor(parsed["video"], tf.float32)

    video = tf.reshape(video, shape=[n_frames, width, height, 3])

    return video, label


def format_features(
    video: tf.Tensor,
    label: str,
    resolution: int = 172,
    scaling_factor: float = 255.0,
    num_classes: int = 2,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Transforms the data to have the appropriate shape.

    This function must be called on a tf.data.Dataset (passed
    via its .map method).

    Args:
        video (tf.Tensor): Decoded video.
        label (str): Corresponding class of the video.
        resolution (int, optional):
            The resolution will be model dependent.
            Movinet a0 and a1 use 172, a2 uses 224.
            Defaults to 172.
        scaling_factor (float, optional):
            Given the videos have the pixels in the range 0.255,
            transforms the data to the range [0, 1]. Defaults to 255..
        num_classes (int, optional):
            Number of classes the model is trained on.
            I.e. for Kinetics 600 will be 600, for UCF 101 that will be the number.
            Defaults to 2.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            When iterated, the first element will be the video, and
            the second will be the label as required by the model.

    """
    label = tf.cast(label, tf.int32)
    label = encode_label(label, num_classes)

    video = tf.image.resize(video, (resolution, resolution))
    video = tf.cast(video, tf.float32) / scaling_factor

    return video, label


def add_states(
    video, label, stream_states={}
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """This function is expected to modify the dataset to make it ready
    for the movinet stream models, but couldn't get to train them

    Args:
        video (_type_): _description_
        label (_type_): _description_
        stream_states (dict, optional): _description_. Defaults to {}.

    Returns:
        Tuple[Dict[str, tf.Tensor], tf.Tensor]: _description_
    """
    return {**stream_states, "image": video}, label


def get_dataset(filenames: List[str]) -> tf.data.Dataset:
    """Generates a td.data.Dataset from the TFRecord files.

    This is the appropriate format to be passed to model.fit,
    after it is formated and there is some batch called, so the
    final video object ingested by the model will have the shape
    [n_videos, n_frames, resolution, resolution, channels].

    Args:
        filenames (List[str]): List of .tfrecord files.

    Returns:
        tf.data.Dataset: Dataset ready to train the model.

    Example:
        target_path is the path to the .tfrecords files directory.

        >>> ds = get_dataset(list(Path(target_path).iterdir()))

        This iterable may be formatted appropriately:

        >>> ds = get_dataset(list(Path(target_path_train).iterdir()))
        >>> ds = ds.map(format_features)

        To see a single example:

        >>> next(iter(ds))
    """
    raw_dataset = tf.data.TFRecordDataset(filenames, compression_type="GZIP")
    return raw_dataset.map(_parse_example)
