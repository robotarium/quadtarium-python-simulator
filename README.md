# quadcopter-simulator-python
Robotarium compatible simulator made in python (Python3.5 or later)

To use this simulator, we recommend using an Anaconda environment ([Anaconda](https://www.anaconda.com/distribution/))

Use the following steps to access the Robotarium examples (after installing anaconda):
1.  **conda create --name [name of environment] --file quad_sim_env.txt** (this will create a python environment with the
necessary package requirements)
1. **conda activate [name of environment]** (activate the environment)
1. **pip install control** (install the python-control through pip)
1. **cd quadcopter-simulator-python/**
1. **python3 -m examples.[filename]** (run the files in the examples/ directory using this command)

To run your own simulations using this simulator:
1. **move your file to the quadcopter-simulation-python/ directory**
1. **make sure it has similar syntax to the example files**
1. run **python3 [your file]**

If you choose to use the simulator without using conda, or the conda text file, download the following packages:
* python=3.5
* numpy=1.15.2
* matplotlib=3.0
* cvxopt=1.2
* control=0.8.2


