"""Help functions. """

from pathlib import Path
from typing import *

import cv2
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio


def load_video_tf(path: str) -> tf.Tensor:
    """Loads a video from an mp4 file and returns the decoded tensor.

    Args:
        path (str): _description_

    Returns:
        tf.Tensor: _description_
    """
    video = tf.io.read_file(path)
    return tfio.experimental.ffmpeg.decode_video(video)


def get_chunks(l: List[Union[str, Path]], n: int) -> Iterable[Union[str, Path]]:
    r"""Yield successive n-sized chunks from l.

    Used to create n sublists from a list l.
    copied
     from: https://github.com/ferreirafabio/video2tfrecord/blob/7aa2c6312e2bc97baed7386b8c92c591769ee5bb/video2tfrecord.py#L55

    Args:
        l (List[Union[str, Path]]):
        n (int):

    Returns:
        Iterable[Union[str, Path]]:

    Example:

        >>> filenames_split = list(get_chunks(filenames, n_videos_in_record))

        To obtain the files from a dataframe.

        >>> next(get_chunks(dataset_df[["classes", "files"]].to_records(index=False), 10))
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


def get_frame_count(path: Path) -> int:
    """Counts the total number of frames in a video.

    Args:
        path (Path): _description_

    Returns:
        int: _description_

    Raises:
        AssertionError: if the path doesn't exist or it couldn't be loaded
            by some reason.

    Example:

        >>> get_video_capture_and_frame_count(sample_clip)
        17
    """
    assert path.is_file(), "Couldn't find video file:" + path + ". Skipping video."

    cap = None

    if path:
        cap = cv2.VideoCapture(str(path))

    assert cap is not None, "Couldn't load video capture:" + path + ". Skipping video."

    # compute meta data of video
    if hasattr(cv2, "cv"):
        frame_count = int(cap.get(cv2.cv.CAP_PROP_FRAME_COUNT))
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return frame_count


def get_label_from_video_name(filename: Path) -> str:
    """Extracts the label of the movement from the video filename.

    Examples:

        >>> path = PosixPath('.../chest-to-bar_6.mp4')
        >>> get_label_from_video_name(path)
        "chest-to-bar"
    """
    return filename.stem.split("_")[0]


def get_labels(filenames: List[Path]) -> List[str]:
    return [get_label_from_video_name(f) for f in filenames]


def create_class_map(path: Union[str, Path]) -> Dict[str, int]:
    """Given a path to a labels.txt file, creates a class map.

    Helper function to obtain the classes for the labels.

    Args:
        path (str or Path): Path pointing to the labels.txt file.

    Returns:
        Dict[str, int]: Dict mapping from labels to classes.
    """
    if isinstance(path, str):
        path = Path(path)
    return {l: i for i, l in enumerate(path.read_text().split("\n"))}


def split_train_test(
    dataset: pd.DataFrame, train_size: float = 0.8, seed: int = 5678
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple function to split a dataset in train/test.

    This functionality may be obtained from many other libraries,
    its just here for personal convinience.

    Args:
        dataset (pd.DataFrame):
            DataFrame with 3 columns: labels, files and classes.
        train_size (float):
            Percentage of the sample for training, range [0, 1].
        seed (int, optional): Random seed to split the data.
            Defaults to 5678.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: tran and test datasets.
    """
    train = dataset.sample(int(len(dataset) * train_size), random_state=seed)
    test = dataset.loc[set(dataset.index).difference(train.index), :]
    return train, test


def get_number_of_steps(samples: int, batch_size: int, epochs: int = 1) -> Tuple[int, int]:
    """Obtain the number of steps.

    Computes the number of steps per epoch and the total number of steps
    to be applied on the LearningScheduler (if used).

    To be called both for train and validation/test.

    Args:
        samples (int): Number of examples in the dataset.
            If the data is splitted in train/test, each of them
            should be treated independently.
        batch_size (int): Number of videos per step.
        epochs (int, optional):
            Number of epochs. If bigger than 1, the second
            returned value corresponds to the total number of
            steps the network will do.
            Defaults to 1.

    Returns:
        Tuple[int, int]: The first argument will either be the number of steps
            per epoch or the number of validation steps when calling `fit` on
            the model, and the second may be used to estimate the learning
            rate scheduler (when number of epochs > 1 and computing steps
            for the training samples).
    """
    steps = samples // batch_size
    return steps, steps * epochs
    
