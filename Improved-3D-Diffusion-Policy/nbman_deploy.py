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

zenoh_path="/home/gr1p24ap0049/projects/gr1-dex-real/teleop-zenoh"
sys.path.append(zenoh_path)
from communication import *
from retarget import ArmRetarget


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

class CommunicationNode(Node):
    def __init__(self):
        super().__init__('idp3_communication_node')

        # 定义 QoS 策略为 BEST_EFFORT
        qos = QoSProfile(
            depth=1,  # 队列深度
            reliability=QoSReliabilityPolicy.BEST_EFFORT  # 设置为尽力而为
        )
        self.bridge = CvBridge()
        
        self.left_arm_publisher_ = self.create_publisher(PoseStamped, "/left_arm_ik_controller/commands", 10)
        self.right_arm_publisher_ = self.create_publisher(PoseStamped, "/right_arm_ik_controller/commands", 10)
        self.left_hand_publisher_ = self.create_publisher(Float64MultiArray, "/left_hand_controller/commands", 10)
        self.right_hand_publisher_ = self.create_publisher(Float64MultiArray, "/right_hand_controller/commands", 10)


        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos)
        
        self.image_sub = self.node.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            qos)
        
        self.pointcloud_sub = self.node.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.pointcloud_callback,
            qos)
        
        self.latest_joint_states = None
        self.latest_image = None
        self.latest_pointcloud = None

    def joint_state_callback(self, msg):
        
        right_hand_joint = np.zeros((1, 6), dtype=np.float32)
        left_hand_joint = np.zeros((1, 6), dtype=np.float32)
        right_arm_joint = np.zeros((1, 6), dtype=np.float32)
        left_arm_joint = np.zeros((1, 6), dtype=np.float32)

        for name in msg.name:
            if "right_joint1" in name:
                right_arm_joint[0][0] = msg.position[msg.name.index(name)]
            if "right_joint2" in name:
                right_arm_joint[0][1] = msg.position[msg.name.index(name)]
            if "right_joint3" in name:
                right_arm_joint[0][2] = msg.position[msg.name.index(name)]
            if "right_joint4" in name:
                right_arm_joint[0][3] = msg.position[msg.name.index(name)]
            if "right_joint5" in name:
                right_arm_joint[0][4] = msg.position[msg.name.index(name)]
            if "right_joint6" in name:
                right_arm_joint[0][5] = msg.position[msg.name.index(name)]
            
            if "left_joint1" in name:
                left_arm_joint[0][0] = msg.position[msg.name.index(name)]
            if "left_joint2" in name:
                left_arm_joint[0][1] = msg.position[msg.name.index(name)]
            if "left_joint3" in name:
                left_arm_joint[0][2] = msg.position[msg.name.index(name)]
            if "left_joint4" in name:
                left_arm_joint[0][3] = msg.position[msg.name.index(name)]
            if "left_joint5" in name:
                left_arm_joint[0][4] = msg.position[msg.name.index(name)]
            if "left_joint6" in name:
                left_arm_joint[0][5] = msg.position[msg.name.index(name)]
            
            if "left_hand_thumb_rotation" in name:
                left_hand_joint[0][0] = msg.effort[msg.name.index(name)]
            if "left_hand_thumb_bend" in name:
                left_hand_joint[0][1] = msg.effort[msg.name.index(name)]
            if "left_hand_index" in name:
                left_hand_joint[0][2] = msg.effort[msg.name.index(name)]
            if "left_hand_middle" in name:
                left_hand_joint[0][3] = msg.effort[msg.name.index(name)]
            if "left_hand_ring" in name:
                left_hand_joint[0][4] = msg.effort[msg.name.index(name)]
            if "left_hand_pinky" in name:
                left_hand_joint[0][5] = msg.effort[msg.name.index(name)]
            
            if "right_hand_thumb_rotation" in name:
                right_hand_joint[0][0] = msg.effort[msg.name.index(name)]
            if "right_hand_thumb_bend" in name:
                right_hand_joint[0][1] = msg.effort[msg.name.index(name)]
            if "right_hand_index" in name:
                right_hand_joint[0][2] = msg.effort[msg.name.index(name)]
            if "right_hand_middle" in name:
                right_hand_joint[0][3] = msg.effort[msg.name.index(name)]
            if "right_hand_ring" in name:
                right_hand_joint[0][4] = msg.effort[msg.name.index(name)]
            if "right_hand_pinky" in name:
                right_hand_joint[0][5] = msg.effort[msg.name.index(name)]
        
            state = np.concatenate([right_arm_joint, left_arm_joint, right_hand_joint, left_hand_joint], axis=0)
            self.latest_joint_states = state

    def image_callback(self, msg):
        # 解析 Image 消息为 numpy 数组
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.latest_image = np.array(self.latest_image, dtype=np.uint8)

    def pointcloud_callback(self, msg):
        # 解析 PointCloud2 消息为 xyzrgb 格式的 numpy 数组
        points_list = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z, rgb = point
            r = (rgb & 0x00FF0000) >> 16
            g = (rgb & 0x0000FF00) >> 8
            b = (rgb & 0x000000FF)
            points_list.append([x, y, z, r, g, b])
        self.latest_pointcloud = np.array(points_list, dtype=np.float32)

    def spin_once(self):
        rclpy.spin_once(self.node)

    def publish_arm_pose(self, right_hand_pose=None, left_hand_pose=None):
        if right_hand_pose is not None:
            right_hand_pose_msg = PoseStamped()
            right_hand_pose_msg.pose.position.x = right_hand_pose[0]
            right_hand_pose_msg.pose.position.y = right_hand_pose[1]
            right_hand_pose_msg.pose.position.z = right_hand_pose[2]
            right_hand_pose_msg.pose.orientation.x = right_hand_pose[3]
            right_hand_pose_msg.pose.orientation.y = right_hand_pose[4]
            right_hand_pose_msg.pose.orientation.z = right_hand_pose[5]
            right_hand_pose_msg.pose.orientation.w = right_hand_pose[6]
            right_hand_pose_msg.header.frame_id = "user_head"
            right_hand_pose_msg.header.stamp = self.get_clock().now().to_msg()
            self.right_arm_publisher_.publish(right_hand_pose_msg)

        if left_hand_pose is not None:
            left_hand_pose_msg = PoseStamped()
            left_hand_pose_msg.pose.position.x = left_hand_pose[0]
            left_hand_pose_msg.pose.position.y = left_hand_pose[1]
            left_hand_pose_msg.pose.position.z = left_hand_pose[2]
            left_hand_pose_msg.pose.orientation.x = left_hand_pose[3]
            left_hand_pose_msg.pose.orientation.y = left_hand_pose[4]
            left_hand_pose_msg.pose.orientation.z = left_hand_pose[5]
            left_hand_pose_msg.pose.orientation.w = left_hand_pose[6]
            left_hand_pose_msg.header.frame_id = "user_head"
            left_hand_pose_msg.header.stamp = self.get_clock().now().to_msg()
            self.left_arm_publisher_.publish(left_hand_pose_msg)

    def publish_hand_action(self, right_qpos=None, left_qpos=None):
        if right_qpos is not None:
            right_hand_msg = Float64MultiArray()
            right_pinky = (1.7 - right_qpos[4]) * 1000 / 1.7
            right_ring = (1.38 - right_qpos[6]) * 1000 / 1.38
            right_middle = (1.13 - right_qpos[2]) * 1000 / 1.13
            right_index = (1.12 - right_qpos[0]) * 1000 / 1.12
            right_thumb_string = (0.6 - right_qpos[8]) * 1000 / 0.6
            right_thumb_motor = (0.42 - right_qpos[9]) * 1000 / 0.42
            right_hand_msg.data = [right_pinky, right_ring, right_middle, right_index, right_thumb_motor, right_thumb_string]
            self.right_hand_publisher_.publish(right_hand_msg)

        if left_qpos is not None:
            left_hand_msg = Float64MultiArray()
            left_pinky = (1.7 - left_qpos[4]) * 1000 / 1.7
            left_ring = (1.38 - left_qpos[6]) * 1000 / 1.38
            left_middle = (1.13 - left_qpos[2]) * 1000 / 1.13
            left_index = (1.12 - left_qpos[0]) * 1000 / 1.12
            left_thumb_string = (0.6 - left_qpos[8]) * 1000 / 0.6
            left_thumb_motor = (0.42 - left_qpos[9]) * 1000 / 0.42
            left_hand_msg.data = [left_pinky, left_ring, left_middle, left_index, left_thumb_motor, left_thumb_string]
            self.left_hand_publisher_.publish(left_hand_msg)
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
        self.communication=CommunicationNode()
        
        
    
    def step(self, action_list):
        
        for action_id in range(self.action_horizon):
            act = action_list[action_id]
            self.action_array.append(act)
            
            filtered_act = act.copy()
            filtered_armpos = filtered_act[:14] #四元数
            filtered_handpos = filtered_act[-12:]
          
            self.communication.publish_arm_pose(filtered_armpos[6:], filtered_armpos[:6])#控制双臂
            self.communication.publish_hand_action(filtered_handpos[6:], filtered_handpos[:6])#控制双手
            
            
            self.communication.spin_once()#获取最新数据
            self.cloud_array.append(self.communication.latest_pointcloud)
            self.color_array.append(self.communication.latest_image)
            # self.depth_array.append(self.communication.latest_depth)
      
            states=self.communication.latest_joint_states()
           
            self.env_qpos_array.append(states)
            
        
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
        # init buffer
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []
    
    
        # pos init
        qpos_init1 = np.array([-np.pi / 12, 0, 0, -1.6, 0, 0, 0, 
            -np.pi / 12, 0, 0, -1.6, 0, 0, 0])
        qpos_init2 = np.array([-np.pi / 12, 0, 1.5, -1.6, 0, 0, 0, 
                -np.pi / 12, 0, -1.5, -1.6, 0, 0, 0])
        hand_init = np.ones(12)
        # hand_init = np.ones(12) * 0

        

        upbody_initpos = np.concatenate([qpos_init1])
        self.upbody_comm.init_set_pos(upbody_initpos)
        q_14d = upbody_initpos.copy()
            
        body_action = np.zeros(6)
        
        # this is a must for eef pos alignment
        arm_pos, arm_rot_quat = action_util.init_arm_pos, action_util.init_arm_quat
        q_14d = self.arm_solver.ik(q_14d, arm_pos, arm_rot_quat)
        self.upbody_comm.init_set_pos(q_14d)
        time.sleep(2)
        
        print("Robot ready!")
        
        # ======== INIT ==========
        # camera.start()
        cam_dict = self.camera()
        self.color_array.append(cam_dict['color'])
        self.depth_array.append(cam_dict['depth'])
        self.cloud_array.append(cam_dict['point_cloud'])

        try:
            hand_qpos = self.hand_comm.get_qpos()
        except:
            cprint("fail to fetch hand qpos. use default.", "red")
            hand_qpos = np.ones(12)

        env_qpos = np.concatenate([self.upbody_comm.get_pos(), hand_qpos])
        self.env_qpos_array.append(env_qpos)
                        
        self.q_14d = q_14d
        self.body_action = body_action
    

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
            
        self.spin_once()
        return obs_dict


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
    record_data = True

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

    # if record_data:
    #     import h5py
    #     root_dir = "/home/gr1p24ap0049/projects/gr1-learning-real/"
    #     save_dir = root_dir + "deploy_dir"
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     record_file_name = f"{save_dir}/demo.h5"
    #     color_array = np.array(env.color_array)
    #     depth_array = np.array(env.depth_array)
    #     cloud_array = np.array(env.cloud_array)
    #     qpos_array = np.array(env.qpos_array)
    #     with h5py.File(record_file_name, "w") as f:
    #         f.create_dataset("color", data=np.array(color_array))
    #         f.create_dataset("depth", data=np.array(depth_array))
    #         f.create_dataset("cloud", data=np.array(cloud_array))
    #         f.create_dataset("qpos", data=np.array(qpos_array))
        
    #     choice = input("whether to rename: y/n")
    #     if choice == "y":
    #         renamed = input("file rename:")
    #         os.rename(src=record_file_name, dst=record_file_name.replace("demo.h5", renamed+'.h5'))
    #         new_name = record_file_name.replace("demo.h5", renamed+'.h5')
    #         cprint(f"save data at step: {roll_out_length} in {new_name}", "yellow")
    #     else:
    #         cprint(f"save data at step: {roll_out_length} in {record_file_name}", "yellow")


if __name__ == "__main__":
    main()
