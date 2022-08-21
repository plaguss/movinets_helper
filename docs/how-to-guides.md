## How to create a dataset.

It is assumed the videos are in mp4 format, but should be similar for other formats.

Assuming the videos/clips are stored with the following layout:

```
clips/
    label_a/
        label_a_1.mp4
        label_a_2.mp4
        ...
    label_b/
        label_b_1.mp4
        label_b_2.mp4
        ...
    ...
```

Keep the labels in a .txt.

```
label_a
label_b
...
```

Grab the paths to the videos and extract the labels:

```Python
import glob
from pathlib import Path
from movinets_helper.utils import get_labels

glob_videos = glob.glob(str(Path(clips_path) / "**/*.mp4"))
video_paths = [Path(p) for p in glob_videos]
labels = get_labels(video_paths)
```

Create a dataset with the paths, labels and classes: 

```python
import pandas as pd
from movinets_helper.utils import create_class_map

class_map = create_class_map("labels.txt")

dataset_df = pd.DataFrame({"labels": labels, "files": glob_videos})
dataset_df["classes"] = dataset_df["labels"].map(class_map)
```

Split the dataset in train and test prior to the TFRecords generation.

```python
from movinets_helper.utils import split_train_test

train_dataset_df, test_dataset_df = split_train_test(dataset_df, train_size=0.8)
```

We are ready to generate the dataset.

*For the number of videos to store per record, see [tfrecord guide](https://www.tensorflow.org/tutorials/load_data/tfrecord).*

*Regarding the number of frames and the resolution, it will be user dependent.*

```py
import movinets_helper.writer as wr

wr.convert_mp4_to_tfrecord(
    train_dataset_df[["classes", "files"]],
    "path-for-training",
    n_videos_in_record: int = 25,
    n_frames_per_video: int = 10,
    resolution: int = 224,
)

wr.convert_mp4_to_tfrecord(
    test_dataset_df[["classes", "files"]],
    "path-for-testing",
    n_videos_in_record: int = 25,
    n_frames_per_video: int = 10,
    resolution: int = 224,
)

```

*Note*: To create a dataset of approximately 540 videos,
with shapes (10, 224, 224, 3), took close to 4 minutes.

Keep in mind, the result files will be compressed with gzip,
but will take up a lot of space.

## How to ingest a dataset.

Access to the dataset generated.

```python
from pathlib import Path
from movinets_helper.reader import get_dataset, format_features

dataset_dir = "path-to-files"
train_dataset_dir = dataset_dir / "train"
test_dataset_dir = dataset_dir / "test"

ds_train = get_dataset(list(Path(train_dataset_dir).iterdir()))
ds_test = get_dataset(list(Path(test_dataset_dir).iterdir()))
```

To make it usable for the model selected, the inputs
must be formatted appropriately and batched.

```python
batch_size = 8  # From the tutorial

ds_train = ds_train.map(
    format_features,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(batch_size)
ds_train = ds_train.repeat()
ds_train = ds_train.prefetch(2)

ds_test = ds_test.map(
    format_features,
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True
).batch(batch_size)
ds_test = ds_test.repeat()
ds_test = ds_test.prefetch(2)
```

The function `format_features` is by default set to the resolution of `a0` model, it may be updated in the following way,
and according to your number of classes:

```python
from functools import partial
format_features_a2 = partial(format_features, resolution=224, num_classes=9)
```

## Fine-Tuning Movinet A2 Base

This package has been used tested on google colab and the version dependencies are adapted for the case.

The `movinet_tutorial.ipynb` uses tensorflow versions
2.9 and higher, and the correct movinet model versions are defined there, but there may be some error when calling `fit` to train the model. In that case, take a look at this [issue](https://github.com/tensorflow/models/issues/10590)

First load the pretrained weights of the chosen model

```bash
!wget https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a0_base.tar.gz -O movinet_a0_base.tar.gz -q

!tar -xvf movinet_a0_base.tar.gz
```

Get the parameters expected for the model. Read the docs for more info on this

```python
import movinet_helper.modelling as model
cfg = model.ConfigMovinetA2Base()
```

KEEP ON THIS POINT

## Load the saved model

Once the model has been fine tuned, it can be loaded to return predictions.


```python
from movinet_helper.utils import load_model
model = load_model("<path_to_model>")
```
