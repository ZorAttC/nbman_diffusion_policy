import zarr
import numpy as np

# 示例数据字典
data = {
    'timestamps': [],
    'states': [],  # (ts, concat[right_arm_joint, left_arm_joint, right_hand_joint, left_hand_joint] 6,6,6,6)
    'actions': [], # (ts, concat[right_arm_action, left_arm_action, right_hand_action, left_hand_action] 7,7,6,6)
    'images': [],  # (ts, image_array), uint8
    'pointclouds': []  # (ts, pointcloud_array), float32，每帧点数可变
}

# 创建示例数据
timestamps = np.array(np.arange(1000))
states = [(t, np.concatenate([np.random.rand(6), np.random.rand(6), 
                             np.random.rand(6), np.random.rand(6)])) for t in timestamps]  # 24个元素
actions = [(t, np.concatenate([np.random.rand(7), np.random.rand(7), 
                              np.random.rand(6), np.random.rand(6)])) for t in timestamps]  # 26个元素
images = [(t, (np.random.rand(224, 224, 3) * 255).astype(np.uint8)) for t in timestamps]  # uint8, 0-255
pointclouds = [(t, np.random.rand(np.random.randint(8000, 12000), 3).astype(np.float32)) for t in timestamps]  # float32

# 更新数据字典
data['timestamps'] = timestamps
data['states'] = states
data['actions'] = actions
data['images'] = images
data['pointclouds'] = pointclouds

# 创建 zarr 文件
store = zarr.DirectoryStore('complex_data.zarr')
root = zarr.group(store=store, overwrite=True)

# 存储 timestamps
root.array('timestamps', data['timestamps'], chunks=(1000,), dtype='float64')

# 存储 states
states_timestamps = np.array([s[0] for s in data['states']])
states_data = np.array([s[1] for s in data['states']])  # shape: (n_samples, 24)
root.create_group('states')
root['states'].array('timestamps', states_timestamps, chunks=(1000,), dtype='float64')
root['states'].array('data', states_data, chunks=(1000, 24), dtype='float64')

# 存储 actions
actions_timestamps = np.array([a[0] for a in data['actions']])
actions_data = np.array([a[1] for a in data['actions']])  # shape: (n_samples, 26)
root.create_group('actions')
root['actions'].array('timestamps', actions_timestamps, chunks=(1000,), dtype='float64')
root['actions'].array('data', actions_data, chunks=(1000, 26), dtype='float64')

# 存储 images (uint8)
images_timestamps = np.array([img[0] for img in data['images']])
images_data = np.array([img[1] for img in data['images']], dtype=np.uint8)  # shape: (n_samples, 224, 224, 3)
root.create_group('images')
root['images'].array('timestamps', images_timestamps, chunks=(1000,), dtype='float64')
root['images'].array('data', images_data, chunks=(100, 224, 224, 3), dtype='uint8')

# 存储 pointclouds (float32，每帧点数可变)
pointclouds_timestamps = np.array([pc[0] for pc in data['pointclouds']])
pointclouds_group = root.create_group('pointclouds')
pointclouds_group.array('timestamps', pointclouds_timestamps, chunks=(1000,), dtype='float64')
for i, (ts, pc) in enumerate(data['pointclouds']):
    pointclouds_group.array(f'data_{i}', pc, chunks=(None, 3), dtype='float32')

# 读取示例
print("Reading back some data:")
loaded = zarr.open('complex_data.zarr', mode='r')
print("Timestamps:", loaded['timestamps'][:])
print("First state data:", loaded['states/data'][0])  # 24 elements
print("First action data:", loaded['actions/data'][0])  # 26 elements
print("First image timestamp:", loaded['images/timestamps'][0])
print("First image shape and dtype:", loaded['images/data'][0].shape, loaded['images/data'][0].dtype)
print("First pointcloud timestamp:", loaded['pointclouds/timestamps'][0])
print("First pointcloud shape and dtype:", loaded['pointclouds/data_0'].shape, loaded['pointclouds/data_0'].dtype)