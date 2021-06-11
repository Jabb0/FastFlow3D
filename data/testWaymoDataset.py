from data.WaymoDataset import WaymoDataset
import os
from utils.plot import visualize_point_cloud, plot_pillars, plot_2d_point_cloud

train_path = './train'
arr = os.listdir(train_path)
waymo_dataset = WaymoDataset(train_path)
print(len(waymo_dataset))

counter = 0
for data in waymo_dataset:
    counter += 1
    print(counter)

kk = 0
#dataset = WaymoDataset()