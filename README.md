# quadcopter-simulator-python
Robotarium compatible simulator made in python (Python3.5 or later).

### Creating a conda environment
To use this simulator, we recommend using an Anaconda environment ([Anaconda](https://www.anaconda.com/distribution/))
.
1.  To create a conda environment with the necessary package requirements:\
    ``conda create --name [name of environment] --file requirements_[platform].txt --channel conda-forge``  
    The supported platforms are `linux-64`, `osx-64` and `win-64`.
1. Activate the environment by running:\
   ``conda activate [name of environment]``
1. Install the python-control through pip:\
    ``pip install control``
1. Navigative to repo:\
    ``cd quadcopter-simulator-python``
1. As a sanity check, run the files in the ``examples`` directory using this command:\
``python -m examples.[filename]`` \
Make sure to omit the `.py`.

### Coding up your own experiment using this simulator
1. Move your file to the `quadcopter-simulation-python` directory
1. Make sure it has similar syntax to the example files (`examples/go_to_point.py` is a good starting point)
1. run `python [your file]`
