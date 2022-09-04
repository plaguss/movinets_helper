"""Help functions. """

from pathlib import Path
from typing import *
import base64

import cv2
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_hub as hub


def load_video_tf(path: str) -> tf.Tensor:
    """Loads a video from an mp4 file and returns the decoded tensor.

    Args:
        path (str): Path to a video.

    Returns:
        tf.Tensor: Tensorflow's representation of the video.
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
    

def load_model(path_to_model: Union[str, Path]) -> "tf.keras.engine.functional.Functional":
    r"""Load a model 

    Args:
        path_to_model (Union[str, Path]): Path to the SavedModel directory.

    Returns:
        tf.keras.engine.functional.Functional: Model ready to predict.
    """
    if isinstance(path_to_model, Path):
        path_to_model = str(path_to_model)

    keras_layer = hub.KerasLayer(path_to_model)

    inputs = tf.keras.layers.Input(
        shape=[None, None, None, 3],
        dtype=tf.float32
    )
    inputs = dict(image=inputs)
    outputs = keras_layer(inputs)

    model = tf.keras.Model(inputs, outputs)
    model.build([1, 1, 1, 1, 3])

    return model


def prepare_to_predict(video: tf.Tensor, resolution: int = 224) -> tf.Tensor:
    """Adds a dimension to a video (and possibly resizes to the wanted resolution)

    The model expects the videos with 5 dimensions. When a video
    was loaded with load_video_tf it will have only the 4 dimension
    expected, which would represent the batch size during training.

    Args:
        video (tf.Tensor): video as tf.Tensor, maybe loaded with load_video_tf.
        resolution (int): 
            If is different from None, resizes the video, to the expected
            resolution of the model. If set to None, doesn't do anything.
            Defaults to 224 (expected shape for a2 model).

    Returns:
        tf.Tensor: video ready to be passed to model.predict(video)
    """
    if resolution is not None:
        video = tf.image.resize(video, [resolution, resolution])
    
    return tf.expand_dims(video, 0)


def get_top_k(probs, k=5, label_map=tf.constant([""])):
    """Outputs the top k model labels and probabilities on the given video."""
    top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
    top_labels = tf.gather(label_map, top_predictions, axis=-1)
    top_labels = [label.decode('utf8') for label in top_labels.numpy()]
    top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
    return tuple(zip(top_labels, top_probs))


def predict_top_k(model, video, k=5, label_map=tf.constant([""])):
    """Outputs the top k model labels and probabilities on the given video.
    
    Args:
        model (_type_): _description_
        video (_type_): _description_
        k (int, optional): _description_. Defaults to 5.
        label_map (_type_, optional): _description_. Defaults to tf.constant([""]).

    Returns:
        _type_: _description_
    """
    outputs = model.predict(video[tf.newaxis])[0]
    probs = tf.nn.softmax(outputs)
    return get_top_k(probs, k=k, label_map=label_map)


def parse_video_from_bytes(video_encoded: bytes) -> tf.Tensor:
    """Parses a base64 encoded video to a tf.Tensor.

    Used to read a video sent through a post request

    Args:
        video_encoded (_type_): base64 video encoded.
            This would be equivalent to:

            >>> with open(video_path, "rb") as video_file:
            >>>    text = base64.b64encode(video_file.read())
            >>>    video_encoded = base64.b64decode(text)


    Returns:
        tf.Tensor: Video as tf.Tensor.
            Returns the same object that would be obtained from load_video_tf.
    """
    return tfio.experimental.ffmpeg.decode_video(base64.b64decode(video_encoded))
