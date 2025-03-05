import zarr
import numpy as np
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='Read data from a zarr file.')
parser.add_argument('-f', '--file', type=str, required=True, help='Path to the zarr file')
args = parser.parse_args()

# 打开 zarr 文件
loaded = zarr.open(args.file, mode='r')

# 读取 states
states_timestamps = loaded['states/timestamps'][:]
if states_timestamps.size > 0:
    states_data = loaded['states/data'][:]  # shape: (n_samples, 24)
    states = [(ts, data) for ts, data in zip(states_timestamps, states_data)]
    print("First state (timestamp, data):", states[0])
else:
    states = []
    print("No states stored")

# 读取 actions
actions_timestamps = loaded['actions/timestamps'][:]
if actions_timestamps.size > 0:
    actions_data = loaded['actions/data'][:]  # shape: (n_samples, 26)
    actions = [(ts, data) for ts, data in zip(actions_timestamps, actions_data)]
    print("First action (timestamp, data):", actions[1:3])
else:
    actions = []
    print("No actions stored")

# 读取 images
images_timestamps = loaded['images/timestamps'][:]
if images_timestamps.size > 0:
    images_data = loaded['images/data'][:]  # shape: (n_samples, 224, 224, 3), uint8
    images = [(ts, data) for ts, data in zip(images_timestamps, images_data)]
    print("First image (timestamp, data shape, dtype):", (images[0][0], images[0][1].shape, images[0][1].dtype))
else:
    images = []
    print("No images stored")

# 读取 pointclouds
pointclouds_timestamps = loaded['pointclouds/timestamps'][:]
if pointclouds_timestamps.size > 0:
    n_frames = len(pointclouds_timestamps)
    pointclouds = [(ts, loaded['pointclouds'][f'data_{i}'][:]) for i, ts in enumerate(pointclouds_timestamps)]
    print("First pointcloud (timestamp, data shape, dtype):", (pointclouds[0][0], pointclouds[0][1].shape, pointclouds[0][1].dtype))
else:
    pointclouds = []
    print("No pointclouds stored")

# 重构数据字典
data_reconstructed = {
    'states': states,
    'actions': actions,
    'images': images,
    'pointclouds': pointclouds
}

# 验证重构数据
print("\nReconstructed data summary:")
print("Number of states:", len(data_reconstructed['states']))
print("Number of actions:", len(data_reconstructed['actions']))
print("Number of images:", len(data_reconstructed['images']))
print("Number of pointclouds:", len(data_reconstructed['pointclouds']))