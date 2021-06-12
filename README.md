# FastFlow3D Implementation
## References
- Jund et al.: Scalable Scene Flow from Point Clouds in the Real World (2021)
- Liu et al.: FlowNet3D: Learning Scene Flow in 3D Point Clouds (2019)
- Dewan et al.: Rigid Scene Flow of 3D Lidar Scans
- Waymo dataset: https://waymo.com/open/


## Architecture


# Template PyTorch Project Repository

## References
- https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_mnist.py
- https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
- https://pytorch-lightning.readthedocs.io/en/latest/starter/converting.html
- https://pytorch-lightning.readthedocs.io/en/latest/benchmarking/performance.html
- https://pytorch-lightning.readthedocs.io/en/latest/starter/style_guide.html
- https://niessner.github.io/I2DL/ Exercise 7

- Do not use .cuda() or .cpu() anymore. Lightning does the handling.

## Usage
Fork this repo to start your own pyTorch based project.

This repo is already structured as a python module such that every part can be used externally too.

## DVC
DVC is initialized but no included into the workflow right now.
Have a look here: https://dvc.org/doc/start

This repository uses Data Version Control (DVC) for management of datasets and results.
DVC does not only allow to have reproducible results but also create a clean experiment process.
The DL workflow can be modularized into different steps and these steps can be chained as desired.

## Structure
### Models
Full PyTorch Lightning modules that are trained.

### Networks
PyTorch modules 

### Data
All dataloading routines and transformations.

