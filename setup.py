from setuptools import setup

setup(
    name='quadcopter-simulator',
    version='0.0.1',
    author='Christopher Banks',
    long_description='',
    description='',
    zip_safe=False,
    packages=['utilities_sim'],
    install_requires=[
        "control",
        "cvxopt",
        "matplotlib",
        "numpy",
    ],
)
