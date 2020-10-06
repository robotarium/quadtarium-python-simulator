# quadtarium-simulator-python
The Robotarium's quadcopter python simulator (Python3.5 or later). 

The purpose of the Robotarium simulator is to ensure that algorithms perform reasonably well before deployment onto the Robotarium's quadcopters (COMING SOON!). The objective is to make scripts created for the simulator directly applicable onto the Robotarium's quadcopters. To ensure minimum modification after deployment, the simulator has been created to closely approximate the actual behavior of the Robotarium's agents.

In terms of user-level control, the user can specify a desired position for each quadcopter at each timestep, which is tracked using a differential flatness controller as in [[1]](#1). A collision avoidance solution based on [[2]](#2) is also provided to the user.

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
1. If you want to install it as a pip package run \
``pip install -e .``

### Coding up your own experiment using this simulator
1. Move your file to the `quadcopter-simulation-python` directory
1. Make sure it has similar syntax to the example files (`examples/go_to_point.py` is a good starting point)
1. run `python [your file]`


### References
<a id="1">[1]</a> 
Mellinger, Daniel, and Vijay Kumar. "Minimum snap trajectory generation and control for quadrotors." 2011 IEEE international conference on robotics and automation. IEEE, 2011. \
<a id="2">[2]</a> 
Wang, Li, Aaron D. Ames, and Magnus Egerstedt. "Safe certificate-based maneuvers for teams of quadrotors using differential flatness." 2017 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2017.
