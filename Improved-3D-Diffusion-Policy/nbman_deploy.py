import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import time
from omegaconf import OmegaConf
import pathlib
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import diffusion_policy_3d.common.gr1_action_util as action_util
import diffusion_policy_3d.common.rotation_util as rotation_util
import tqdm
import torch
import os 
os.environ['WANDB_SILENT'] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


from diffusion_policy_3d.common.multi_realsense import MultiRealSense

import datetime


import numpy as np
import torch
from termcolor import cprint

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState, Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2

from communication_node import CommunicationNode
import zarr

class NbManEnvInference:
    """
    The deployment is running on the Orin AGX of the robot.
    """
    def __init__(self, obs_horizon=2, action_horizon=8, device="gpu",
                use_point_cloud=True, use_image=True, img_size=224,
                 num_points=4096,
                 use_waist=False):
        
        # obs/action
        self.use_point_cloud = use_point_cloud
        self.use_image = use_image
        
        
        # horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # inference device
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        # Communication
        self.communication_node=CommunicationNode()

        # Data buffer init
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []
        
        
    
    def step(self, action_list):
        
        for action_id in range(self.action_horizon):
            act = action_list[action_id]
            self.action_array.append(act)
            
            filtered_act = act.copy()
            filtered_armpos = filtered_act[:14] #四元数
            filtered_handpos = filtered_act[-12:]
          
            self.communication_node.publish_arm_pose(filtered_armpos[6:], filtered_armpos[:6])#控制双臂
            self.communication_node.publish_hand_action(filtered_handpos[5:], filtered_handpos[:5])#控制双手
            
            
            self.communication_node.spin_once()#获取最新数据
            # self.depth_array.append(self.communication_node.latest_depth)
            self.env_qpos_array.append(self.communication_node.get_latest_joint_states())
         
            
        
        agent_pos = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)
    
        obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)
        obs_img = np.stack(self.color_array[-self.obs_horizon:], axis=0)
            
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)
        return obs_dict
    
    def reset(self, first_init=True):
        if self.communication_node is None:
            print("communication node is none")
            return
        # init buffer
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []
    
        # Anchor: pos init
        right_arm_init_pose = np.array([1,2,3,4,5,6,7],dtype=np.float32)
        left_arm_init_pose = np.array([1,2,3,4,5,6,7],dtype=np.float32)
        hand_init = np.ones(6)
        # hand_init = np.ones(12) * 0

        execute_time=3
        while rclpy.ok() and execute_time>0:
            self.communication_node.publish_arm_pose(right_arm_init_pose, left_arm_init_pose)
            self.communication_node.publish_hand_action(hand_init, hand_init)
            time.sleep(0.01)
            execute_time-=0.01
            
        
        print("Robot ready!")
        
        # ======== INIT ==========
        # camera.start()
        self.communication_node.spin_once()
        self.color_array.append(self.communication_node.get_latest_image())
        # self.depth_array.append(communication_node.get_latest())
        self.cloud_array.append(self.communication_node.get_latest_pointcloud())
        self.env_qpos_array.append(self.communication_node.get_latest_joint_states())

     
    

        agent_pos = np.stack([self.env_qpos_array[-1]]*self.obs_horizon, axis=0)
        
        obs_cloud = np.stack([self.cloud_array[-1]]*self.obs_horizon, axis=0)
        obs_img = np.stack([self.color_array[-1]]*self.obs_horizon, axis=0)
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        if self.use_point_cloud:
            obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_dict['image'] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)
            
        self.communication_node.spin_once()
        return obs_dict
    def save_data(self, save_path):
        # 创建 zarr 文件
        store = zarr.DirectoryStore(save_path)
        root = zarr.group(store=store, overwrite=True)
        
        # 存储 states
        root.create_group('states')
        if self.env_qpos_array:
            states_data = np.array(self.env_qpos_array)  # shape: (n_samples, 24)
            root['states'].array('data', states_data, chunks=(1000, 24), dtype='float64')
            print(f"Saved {len(self.env_qpos_array)} states")
        else:
            root['states'].array('data', np.array([], dtype=np.float64).reshape(0, 24), chunks=(1000, 24), dtype='float64')
            print("No states to save, storing empty arrays")

        # 存储 actions
        root.create_group('actions')
        if self.action_array:
            actions_data = np.array(self.action_array)  # shape: (n_samples, 26)
            root['actions'].array('data', actions_data, chunks=(1000, 26), dtype='float64')
            print(f"Saved {len(self.action_array)} actions")
        else:
            root['actions'].array('data', np.array([], dtype=np.float64).reshape(0, 26), chunks=(1000, 26), dtype='float64')
            print("No actions to save, storing empty arrays")

        # 存储 images (uint8)
        root.create_group('images')
        if self.color_array:
            images_data = np.array(self.color_array, dtype=np.uint8)  # shape: (n_samples, 224, 224, 3)
            root['images'].array('data', images_data, chunks=(100, 224, 224, 3), dtype='uint8')
            print(f"Saved {len(self.color_array)} images")
        else:
            root['images'].array('data', np.array([], dtype=np.uint8).reshape(0, 224, 224, 3), chunks=(100, 224, 224, 3), dtype='uint8')
            print("No images to save, storing empty arrays")

        # 存储 pointclouds (float32，每帧点数可变)
        pointclouds_group = root.create_group('pointclouds')
        if self.cloud_array:
            for i, pc in enumerate(self.cloud_array):
                pointclouds_group.array(f'data_{i}', pc, chunks=(None, 3), dtype='float32')
            print(f"Saved {len(self.cloud_array)} pointclouds")
        else:
            print("No pointclouds to save")

        # 日志记录
        print(f"Data saved to {save_path}")

@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d','config'))
)
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    if workspace.__class__.__name__ == 'DPWorkspace':
        use_image = True
        use_point_cloud = False
    else:
        use_image = False
        use_point_cloud = True
        
    # fetch policy model
    policy = workspace.get_model()
    action_horizon = policy.horizon - policy.n_obs_steps + 1

    # pour
    roll_out_length_dict = {
        "pour": 300,
        "grasp": 1000,
        "wipe": 300,
    }
    # task = "wipe"
    task = "grasp"
    # task = "pour"
    roll_out_length = roll_out_length_dict[task]
    
    img_size = 224
    num_points = 4096
    use_waist = True
    first_init = True
    record_data = True #记录实验数据

    env = NbManEnvInference(obs_horizon=2, action_horizon=action_horizon, device="cpu",
                             use_point_cloud=use_point_cloud,
                             use_image=use_image,
                             img_size=img_size,
                             num_points=num_points,
                             use_waist=use_waist)

    
    obs_dict = env.reset(first_init=first_init)

    step_count = 0
    
    while step_count < roll_out_length:
        with torch.no_grad():
            action = policy(obs_dict)[0]
            action_list = [act.numpy() for act in action]
        
        obs_dict = env.step(action_list)
        step_count += action_horizon
        print(f"step: {step_count}")

    if record_data:
       
        save_dir = "deploy_dir"
        os.makedirs(save_dir, exist_ok=True)
        save_name = 'test'+datetime.now().strftime("%Y%m%d_%H%M%S") + ".zarr"
        save_path = os.path.join(save_dir, save_name)
        env.save_data(save_path)
        cprint(f"Data saved to {save_path}", 'green')

if __name__ == "__main__":
    main()
