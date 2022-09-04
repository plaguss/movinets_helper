## General Installation

If you are going to train the model using Google Colab, it may be better to skip this section and go directly to the next one.

Otherwise, to install `movinets_helper`, just pip install it:

```bash
pip install movinets_helper
```


### Installing on Google Colab

When dealing with Google Colab, the preferred way to install this package is to just install the dependencies needed, and after this, install the package without the dependencies.

The dependencies should be the same to the `movinet_tutorial.ipynb` offered by the authors (keep in mind this commands are rin in the notebook):

```bash
!python -m pip install --upgrade pip

!pip install -q tf-models-official
!pip install tensorflow-io
!pip uninstall -q -y opencv-python-headless
!pip install -q "opencv-python-headless<4.3"
!pip install tensorflow==2.8
!apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
```

After the correct installation of these dependencies, install the package:

```bash
pip install --no-deps movinets_helper
```
