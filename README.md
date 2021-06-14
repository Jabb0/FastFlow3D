# FastFlow3D Implementation
Source: https://arxiv.org/pdf/2103.01306.pdf

## References
- Jund et al.: Scalable Scene Flow from Point Clouds in the Real World (2021)
- Liu et al.: FlowNet3D: Learning Scene Flow in 3D Point Clouds (2019)
- Dewan et al.: Rigid Scene Flow of 3D Lidar Scans
- Waymo dataset: https://waymo.com/open/

## Problem Definition
- 3D scene flow in a setting where the scene at time $t_i$ is represented as a point cloud $P_i$ as measured by a LiDAR sensor mounted on the AV.
- Each point cloud point has a 3D motion vector (v_x, v_y, v_z). Each component gives the velocity in m/s.
- Predict the flow given two consecutive point clouds.
- With high frame rate the calculation between two timesteps is a good approximation of the current flow.

## Label Creation
The labels are already created for each point in the pointcloud. The paper states how this is done.
They present a scalable automated approach bootstrapped from existing labeled, tracked objects in the LiDAR data sequences.
Each object is a 3D label bounding box with a unique ID.
Ego movement of the AV is removed by computing the position of each object in the previous timeframe in the current timeframe based on the AV movement since this timeframe.
Not compensating for ego movement has shown to decrease performance significantly.
Afterwards self movement of all the points of an object are computed.
One could also have chosen not to remove the AV ego movement, but: You get movement of the object independent of the AV, better reasoning about the objects own movement.
The movement of whole bounding boxes is used initially to identify the movement of their point cloud points. This includes rotations of the objects.
This rotation and ego movement compensation is combined into a single transition matrix T.
For each point x0 in the bounding box of a object in the current pointcloud the previous point position x-1 is computed based on this transition matrix.
This, however, does not mean that the previous pointcloud actually had a point at this position, it is just a "this point has likely moved from this position and thus has this speed".
Because of the point-wise approximation points in the same object can have different speeds.
The label is then the change over time from the previous to the current point.
This label creation can be applied to any point cloud dataset with 3D bounding boxes and tracklets.

### Limitations
- Objects are assumed to be rigid. This does not yield for pedestrians, but as the reference time is so small this is considered of minimal consequence.
- Objects can not have a previous frame
- Some rare moving objects do not have bounding boxes and are belonging to the "background" class without movement. This needs to be overcome.

## Metrics
- Common metrics are L2 error of the pointswise flow as the label is a 3D vector.
- Points with L2 error below a given threshold.
- Metrics per object class as they have inherently different characteristics
- Binary moving/not-moving classification based on a movement threshold. Threshold of 0.5 m/s is selected but not defined easily. Use standard binary classification metrics.


## Architecture
### Input
- Two subsequent point clouds.

### Scene Encoder
- Performed on each input point cloud
- PointNet encoder https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf
- Dynamic voxelization onto a spatial grid http://proceedings.mlr.press/v100/zhou20a/zhou20a.pdf
- Fixed grid of size 512x512. Each grid cell being a pillar.
ATTENTION: Not sure if this is done this way. The dynamic voxelization paper looks like a whole network to create the grid.
1. Take each input point cloud
2. Compute the center coordinates of all 512x512 pillars
3. Compute the pillar that each point falls into
4. Encode each point as 8D (pillarCenter_x, pillarCenter_y, pillarCenter_z, offset_x, offset_y, offset_y, feature_0, feature_1) with offset being the offset from the point to its pillar center and the features being the laser features of the point.
5. Feed each point into an MLP to compute an embedding
6. Sum up the embeddings for all points in a pillar with depth 64 to get pillar embedding
7. The final point cloud embedding is a 512x512 2D image with depth 64

### Contextual Information Decoder
ATTENTION: Knowledge about the U-net architecture still missing but seems required.
- Convolutional Autoencoder (U-net) with first half of the architecture consists of shared weights across two frames.
- Both inputs have the same weights in the conv net
- Both input frames are concatenated in depth
- UNCLEAR: Are they really concatenated given to the encoder? I think they pass each one to the same encoder with shared weights but concatenation is only done when the data is fed as skip connection to the decoder.
- ANSWER: I think the encoder only sees the information of one image. The decoder then gets a concatenation but unclear how.
- Bottleneck convolutions have been introduced.

### Decoder to obtain Pointwise Flow
- Decoder part of the conv autoencoder
- Deconvolutions are replaced with bilinear upsampling.
- Skip connections from the concatenated encoder embeddings at the corresponding spatial resolution.
- Output is a 512x512 grid-structured flow embedding

### Unpillar Operation
- Select for each point the corresponding cell and the cells flow embedding
- Concatenates the point feature (the same 8D vector as given as input?) of the point to predict (we predict the current timeframe t0 only, not t-1).
- MLP to regress onto point-wise motion prediction


## Implementation Notes
- Objects that not have a previous frame cannot be predicted and this need to be removed from weight updates and scene flow evaluation metrics. Their points are still part of the input but their label is not used for training.
- Apparently the z-axis is the height.
- Using vertical pillarization makes sense as objects can only rotate around the z-axis in the Waymo dataset. Therefore, the a pillar is a good capture of the same points in an object.
- Use mean L2 (MSE) loss for training
- Is the ego motion compensation necessary on the previous frame here?
- They trained their model for 19 epochs using the Adam optimizer.
- They applied an artificiall downweight of background points in the L2 loss by a factor of 0.1. This value is found using hyperparameter search. Maybe redo this with a better searcher.


## Data
These are the points that are considered to be seen be the LiDAR:
170m x 170m grid centered at the AV represented by 512 x 512 pillars (approx. 0.33m x 0.33m pillars).
For height (z-dimensions) a valid pillar range is from -3m to 3m.

#### WaymoDataset
It reads the WaymoDataset with the extended flow information. It can be found [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow). You have to register into Waymo to be able to see it. Then, it should be downloaded into *data/train* and *data/val*, respectively.

Each of the file is a session compressed, which has to be decompressed and parsed to access to their field. A session has a number of frames and in each frame is all the information needed. The information of each frame is available [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto). Note that that may a field cannot be accessed in a direct way, so [these](https://github.com/Jabb0/FastFlowNet3D/blob/main/data/util.py) functions should be extended.

Regarding general information about the fields, we are interesented in the 3D LiDAR points. The car in which this information has been logged had 3D LiDARS sensors and, per each sensor, it records the first and second return:

![Returns illustration](https://desktop.arcgis.com/es/arcmap/10.3/manage-data/las-dataset/GUID-0AE5C4B0-4EF6-43F1-B3EE-DC0BBEED4E9A-web.png)

When calling the _utils_ functions, we take into consideration both returns from the five LiDARs and concatenate all of them, so all of them are treated equally. [main.py](https://github.com/Jabb0/FastFlowNet3D/blob/main/main.py) file includes an example on how to read the data of a frame.

More details: https://waymo.com/open/data/perception/

What is this? https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow
And what is this? https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0


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
