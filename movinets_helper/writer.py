"""Helper to write mp4 videos to TFRecordDataset. """

from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

from .utils import get_chunks, get_frame_count, load_video_tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(label: tf.Tensor, video: tf.Tensor) -> str:
    """Creates a tf.train.Example message ready to be written to a file.

    The output can be sent to a TFRecordWriter to be written.

    Args:
        label (tf.Tensor): int encoded label.
        video (tf.Tensor): video as a tensor, maybe loaded using load_video_tf.

    Returns:
        str: representation of the features to be stored as .tfrecords.

    Note:
        Depending on the size (i.e., videos), jupyter won't be able to show them,
        a javascript error will raise.
    """
    # dict mapping the feature name to the tf.train.Example-compatible
    # data type.
    features = {
        "label": _int64_feature(label),
        "n_frames": _int64_feature(video.shape[0]),
        "width": _int64_feature(video.shape[1]),
        "height": _int64_feature(video.shape[2]),
        "n_channels": _int64_feature(video.shape[3]),
        "video": _bytes_feature(tf.io.serialize_tensor(video.numpy())),
    }
    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def _select_frames(
    video: tf.Tensor, frame_count: int, n_frames_per_video: int
) -> tf.Tensor:
    """Selects the frames in a video"""
    return tf.gather(
        video,
        np.arange(0, frame_count, frame_count / n_frames_per_video).round().astype(int),
        axis=0,
    )


def convert_mp4_to_tfrecord(
    dataset_df: pd.DataFrame,
    target_path: Path,
    n_videos_in_record: int = 15,
    n_frames_per_video: int = 10,
    resolution: int = 224,
):
    r"""Saves videos in mp4 format to tfrecords.

    It creates a new dataset of homogeneus videos, all with shape:
    (n_frames, resolution, resolution, channels=3), along
    with its label and shape features to be loaded back.

    Args:
        dataset_df (pd.DataFrame):
            Dataframe with 2 columns: classes, representing the labels, and files,
            with the path to the videos.
        target_path (Path):
            Path where the files will be generated.
            It must be already generated.
        n_videos_in_record (int, optional):
            Number of videos per file.
            See the notes in https://www.tensorflow.org/tutorials/load_data/tfrecord
            Defaults to 10.
        n_frames_per_video (int, optional):
            Number of frames to select per video.
            Defaults to 10.
        resolution (int, optional):
            Try to keep to as low as possible to save space and computing time during
            writing. This can be updated later.
            Defaults to 224.

    Example:

        Generate a dataset of tf records

        >>> convert_mp4_to_tfrecord(dataset_df[["classes", "files"]], target_path)

        Split into train/test

        >>> convert_mp4_to_tfrecord(train_dataset_df[["classes", "files"]], target_path_train)
        >>> convert_mp4_to_tfrecord(test_dataset_df[["classes", "files"]], target_path_test)

    Note:
        This data format isn't storage friendly, even after compression.
    """
    if not Path(target_path).is_dir():
        print(f"{target_path} doesn't exist, create it first.")
        return

    assert dataset_df.columns.tolist() == [
        "classes",
        "files",
    ], "dataset_df doesn't have the expected columns"
    files = dataset_df.to_records(index=False)  # List of tuples (label, filename)
    filenames_split = list(get_chunks(files, n_videos_in_record))

    total_batches = len(filenames_split)

    for i, batch in enumerate(tqdm.tqdm(filenames_split)):
        tfrecord_filename = str(
            Path(target_path) / f"video_batch_{i}_{total_batches}.tfrecords"
        )
        with tf.io.TFRecordWriter(
            tfrecord_filename, options=tf.io.TFRecordOptions(compression_type="GZIP")
        ) as file_writer:
            for label, file in batch:
                try:
                    video = load_video_tf(file)
                except tf.errors.InvalidArgumentError:
                    print(f"Video without length, passing")
                    continue
                video = tf.image.resize(video, (resolution, resolution))
                frame_count = get_frame_count(Path(file))
                video = _select_frames(video, frame_count, n_frames_per_video)
                try:
                    file_writer.write(serialize_example(label, video))
                except Exception as exc:
                    print(f"Couldn't serialize file: {file}")
                    print(f"err: {exc}")
