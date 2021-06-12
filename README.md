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

#### WaymoDataset
It reads the WaymoDataset with the extended flow information. It can be found [here](http://google.com). You have to register into Waymo to be able to see it. Then, it should be downloaded into *data/train* and *data/val*, respectively.

Each of the file is a session compressed, which has to be decompressed and parsed to access to their field. A session has a number of frames and in each frame is all the information needed. The information of each frame is available [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto). Note that that may a field cannot be accessed in a direct way, so [these](https://github.com/Jabb0/FastFlowNet3D/blob/main/data/util.py) functions should be extended.

Regarding general information about the fields, we are interesented in the 3D LiDAR points. The car in which this information has been logged had 3D LiDARS sensors and, per each sensor, it records the first and second return:

![Returns illustration](https://desktop.arcgis.com/es/arcmap/10.3/manage-data/las-dataset/GUID-0AE5C4B0-4EF6-43F1-B3EE-DC0BBEED4E9A-web.png)

When calling the _utils_ functions, we take into consideration both returns from the five LiDARs and concatenate all of them, so all of them are treated equally. [main.py](https://github.com/Jabb0/FastFlowNet3D/blob/main/main.py) file includes an example on how to read the data of a frame.
