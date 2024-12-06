from setuptools import setup, find_packages

setup(
    name="mnist_pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.1',
        'torchvision>=0.15.2',
        'numpy>=1.24.3',
        'pytest>=7.3.1',
    ],
) 