# FastFlow3D Implementation
This repository contains an implementation of the FastFlow3D architecture from "Scalable Scene Flow from Point Clouds in the Real World (Jund et al. 2021)" in PyTorch (with PyTorch lightning). [Paper on arxiv](https://arxiv.org/abs/2103.01306v1).

As a baseline the FlowNet3D architecture by Liu et al. (2019) is implemented as well.

The repository allows to work with the Waymo dataset and its scene flow annotations as well as the FlyingThings3D dataset.

As of now the documentation is not final. For now most documentation refers to FastFlow3D on the waymo dataset. Find early development notes below and detailed comments in the code.

See [USAGE](#usage) for how to jump right in and [RESULTS](#results) for our results.

# LICENSE
See [LICENSE](https://github.com/Jabb0/FastFlow3D/blob/main/LICENSE)

## Cite

If you use this code in your own work, please use the following bibtex entries:

```bibtex
@misc{fastflow3d-pytorch-2021, 
  title={FastFlow3D-PyTorch: Implementation of the FastFlow3D scene flow architecture (Jund et. al 2021) in PyTorch.}, 
  author={Jablonski, Felix and Distelzweig, Aron and Maranes, Carlos}, 
  year={2021}, publisher={GitHub}, 
  howpublished={\url{https://github.com/Jabb0/FastFlow3D}} }
 ``` 
 
 Please don't forget to cite the underlying papers as well!

# Contributors
This repository is created by [Aron](https://github.com/arndz), [Carlos](https://github.com/cmaranes) and [Felix](https://github.com/Jabb0).

## Bugs / Improvements
If you encounter any bugs or you want to suggest any improvements feel free to create a pull request/issue.

# Results
## Experimental Setup
Trained the network for 19 epochs (3 days) on the full waymo train set (157,283 samples) and validated on the full waymo validation set.  
87% of the points in the Waymo dataset are background points with no flow.

Training is done as close as possible to the original paper but because of hardware limitations the batch size is set to 16 instead of 64.
Loss function is the average L2-error in m/s over all points with downweighting of the points belonging to the background class.

### Waymo Dataset Distribution
The distribution of points per label in the waymo dataset has been analyzed.

|       | Total Samples | Total Points | Unlabeled | Background | Vehicle | Pedestrian | Sign  | Cyclist |
|-------|---------------|--------------|-----------|------------|---------|------------|-------|---------|
| Train | 157,283       | 24,265M      | 1.03%     | 87.36%     | 10.52%  | 0.78%      | 0.28% | 0.03%   |
| Valid | 39,785        | 6,193M       | 1.02%     | 88%        | 9.98%   | 0.71%      | 0.25% | 0.03%   |


## Metrics
Here we present two error metrics from the original paper. Same as Jund et. al we have used grouping based on classes (vehicle, pedestrian, background,...) to show inbalances in the performance.

**mean L2 error in m/s** L2 error between the 3D velocity vector for prediction and target averaged over all points. Lower is better.

**<= 1.0 m/s** Percentage of points that are predicted correctly up to 1.0 m/s. Higher is better.

## Quantitative Results
### Waymo Dataset
Comparison of "our" experiment as describe above using this code against the results reported by Jund et. al.

**Note:** Difference in performance are likely due to the different batch size used (16 vs. 64).

![image](https://user-images.githubusercontent.com/33359018/136538870-27f11117-adfe-4cbe-a901-cc6ac90b8bfa.png)


# Usage
This repository contains different parts: Preprocessing, training and visualization.

The hardware requirements of FastFlowNet are high due to the large size of the pseudo images. Reasonable results can be expected within a few days using 4x NVIDIA Titan X GPUs with 12GB VRAM each.

### Compatibility Note
Some dependencies still require Python 3.8 as the 3.9 versions are not available via pip. This is the case for open3d which is used for visualization.

## Preprocessing
In order to use the data it needs to be preprocessed. During preprocessing the dataset is extracted and the important information is stored directly accessible on disc.
The whole WaymoDataset has 1TB of data, the preprocessed data can be stored in 400GB. It is not necessary to use the full dataset, although recommended for best performance.

Download the waymo dataset as tfrecord files from [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow). You have to register into [Waymo](https://waymo.com/open/) to be able to see it. Then, it should be downloaded into `<raw_data_directory/train` and `<raw_data_directory/valid`, respectively.

Start the preprocessing for train and val (and test) separately using:
```bash
python preprocess.py <raw_directory> <out_directory>
```

The output directory has to have a structure of `<directory>/train` for the training data and `<directory>/valid` for the validation data. If test data is available put it into `<directory>/test`.

## Training
Start an experiment with (using Tensorboard logging):
```bash
python train.py <data_directory> <experiment_name>
```

Have a look at the `run.sh` shell scripts and the available parameters of the train.py script.

This project has been built for usage with Weights & Biases as a logging service, thus logging with WnB is supported via command line arguments.

## Visualization
To create a visualization of the point clouds with ground truth and predicted data use the `visualization.py` script.

```bash
python visualization.py <directory>/valid <config_file> --model_path <path_to_checkpoint>>
```

**NOTE:** the current code requires a WeightsAndBiases `config.yaml` thus this logger needs to be used (or the code adapted).

## Data
These are the points that are considered to be seen be the LiDAR:
170m x 170m grid centered at the AV represented by 512 x 512 pillars (approx. 0.33m x 0.33m pillars).
For height (z-dimensions) a valid pillar range is from -3m to 3m.

#### WaymoDataset
It reads the WaymoDataset with the extended flow information. It can be found [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow).

Each of the file is a session compressed, which has to be decompressed and parsed to access to their field. A session has a number of frames and in each frame is all the information needed. The information of each frame is available [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto). Note that that may a field cannot be accessed in a direct way, so [these](https://github.com/Jabb0/FastFlowNet3D/blob/main/data/util.py) functions should be extended.

Regarding general information about the fields, we are interesented in the 3D LiDAR points. The car in which this information has been logged had 3D LiDARS sensors and, per each sensor, it records the first and second return.

When calling the _utils_ functions, we take into consideration both returns from the five LiDARs and concatenate all of them, so all of them are treated equally. [main.py](https://github.com/Jabb0/FastFlowNet3D/blob/main/main.py) file includes an example on how to read the data of a frame.

More details: https://waymo.com/open/data/perception/

## References
- Jund et al.: [Scalable Scene Flow from Point Clouds in the Real World (2021)](https://arxiv.org/pdf/2103.01306.pdf)
- Liu et al.: FlowNet3D: Learning Scene Flow in 3D Point Clouds (2019)
- Dewan et al.: Rigid Scene Flow of 3D Lidar Scans
- Waymo dataset: https://waymo.com/open/

## Problem Definition
- 3D scene flow in a setting where the scene at time $t_i$ is represented as a point cloud $P_i$ as measured by a LiDAR sensor mounted on the AV.
- Each point cloud point has a 3D motion vector (v_x, v_y, v_z). Each component gives the velocity in m/s.
- Each point also has a label identifying its class. This is mostly used for loss weighting but some metrics depend on it too.
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
For each point x0 in the bounding box of an object in the current pointcloud the previous point position x-1 is computed based on this transition matrix.
This, however, does not mean that the previous pointcloud actually had a point at this position, it is just a "this point has likely moved from this position and thus has this speed".
Because of the point-wise approximation points in the same object can have different speeds.
The label is then the change over time from the previous to the current point.
This label creation can be applied to any point cloud dataset with 3D bounding boxes and tracklets.

### Limitations
- Objects are assumed to be rigid. This does not yield for pedestrians, but as the reference time is so small this is considered of minimal consequence.
- Objects can not have a previous frame
- Some rare moving objects do not have bounding boxes and are belonging to the "background" class without movement. This needs to be overcome.

## Metrics
- Common metrics are L2 error of the point wise flow as the label is a 3D vector.
- Points with L2 error below a given threshold.
- Metrics per object class as they have inherently different characteristics
- Binary moving/not-moving classification based on a movement threshold. Threshold of 0.5 m/s is selected but not defined easily. Use standard binary classification metrics.


## Architecture
### Input
- Two subsequent point clouds. Each cloud is a (N_points, 5) matrix. Each point has 5 features, 3 coordinates and 2 laser features.

### Scene Encoder
- Performed on each input point cloud
- Every point is assigned to a fixed 512x512 grid depending on its x, y positions.
- First each point is encoded depending on its grid cell. Each cell is a pillar in z (upwards) direction.
1. Take each input point cloud
2. Compute the center coordinates of all 512x512 pillars
3. Compute the pillar that each point falls into
4. Encode each point as 8D (pillarCenter_x, pillarCenter_y, pillarCenter_z, offset_x, offset_y, offset_z, feature_0, feature_1) with offset being the offset from the point to its pillar center and the features being the laser features of the point.
5. For each point an embedding is computed based on its 8D encoding using an MLP
7. Sum up the embeddings for all points in a pillar with depth 64 to get pillar embedding
8. The final point cloud embedding is a 512x512 2D pseudo-image with depth 64

This part is not straight forward as each point cloud has a different amount of points and thus all points clouds cannot be batched.
This is solved without dropping any points using a scatter-gather approach in FlowNet3D

### Contextual Information Encoder
- Convolutional Autoencoder (U-net) with first half of the architecture consists of shared weights across two frames.
- Both inputs have the same weights in the conv net
- The frames are not concatenated for the encoder pass. Each frame is passed on its own.
  - However to get the highest possible batch size for BatchNorm previous and current frames are passed concatenated in the batch dimension.
- Bottleneck convolutions have been introduced.
- The encoder consists of blocks that each reduce to a hidden size.
- The output of each hidden size is saved.

### Decoder to obtain Pointwise Flow
- Decoder part of the conv autoencoder
- Deconvolutions are replaced with bilinear upsampling.
- Skip connections from the concatenated encoder embeddings at the corresponding spatial resolution.
- Output is a 512x512 grid-structured flow embedding
- No activation function in this part

### Unpillar Operation
- Select for each point the corresponding cell and the cells flow embedding
- This lookup struggles with the different sized points clouds as well. A gather approach is applied to solve this issue.
- Concatenates the point feature (the 64D embedding) of the point to predict (we predict the current timeframe t0 only, not t-1).
- MLP to regress onto point-wise motion prediction


## Implementation Notes
- Objects that not have a previous frame cannot be predicted and this need to be removed from weight updates and scene flow evaluation metrics. Their points are still part of the input but their label is not used for training.
- The z-axis is the height.
- Using vertical pillarization makes sense as objects can only rotate around the z-axis in the Waymo dataset. Therefore, the a pillar is a good capture of the same points in an object.
- Use mean L2-loss for training calculating the average error in speed over all points.
- Ego motion is removed by a translation matrix that transforms the previous frame to the view of the current frame.
- The original authors trained their model for 19 epochs using the Adam optimizer.
- The original authors applied an artificial downweight of background points in the L2 loss by a factor of 0.1. This value is found using hyperparameter search. Maybe redo this with a better searcher.

# General pyTorch Development Notes

## Tutorials / Technical Information
- https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_mnist.py
- https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
- https://pytorch-lightning.readthedocs.io/en/latest/starter/converting.html
- https://pytorch-lightning.readthedocs.io/en/latest/benchmarking/performance.html
- https://pytorch-lightning.readthedocs.io/en/latest/starter/style_guide.html
- https://niessner.github.io/I2DL/ Exercise 7

- Do not use .cuda() or .cpu() anymore. Lightning does the handling.

## Structure
### Models
Full PyTorch Lightning modules that are trained.

### Networks
PyTorch modules 

### Data
All dataloading routines and transformations.
