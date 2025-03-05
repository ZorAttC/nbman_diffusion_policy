import numpy as np
from termcolor import cprint
import zarr
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState, Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import time

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

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos)
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            qos)
        
        self.pointcloud_sub = self.create_subscription(
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
                left_hand_joint[0][0] = msg.position[msg.name.index(name)]
            if "left_hand_thumb_bend" in name:
                left_hand_joint[0][1] = msg.position[msg.name.index(name)]
            if "left_hand_index" in name:
                left_hand_joint[0][2] = msg.position[msg.name.index(name)]
            if "left_hand_middle" in name:
                left_hand_joint[0][3] = msg.position[msg.name.index(name)]
            if "left_hand_ring" in name:
                left_hand_joint[0][4] = msg.position[msg.name.index(name)]
            if "left_hand_pinky" in name:
                left_hand_joint[0][5] = msg.position[msg.name.index(name)]
            
            if "right_hand_thumb_rotation" in name:
                right_hand_joint[0][0] = msg.position[msg.name.index(name)]
            if "right_hand_thumb_bend" in name:
                right_hand_joint[0][1] = msg.position[msg.name.index(name)]
            if "right_hand_index" in name:
                right_hand_joint[0][2] = msg.position[msg.name.index(name)]
            if "right_hand_middle" in name:
                right_hand_joint[0][3] = msg.position[msg.name.index(name)]
            if "right_hand_ring" in name:
                right_hand_joint[0][4] = msg.position[msg.name.index(name)]
            if "right_hand_pinky" in name:
                right_hand_joint[0][5] = msg.position[msg.name.index(name)]
        
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
        rclpy.spin_once(self)

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
           
            right_hand_msg.data = right_qpos.tolist()
            self.right_hand_publisher_.publish(right_hand_msg)

        if left_qpos is not None:
            left_hand_msg = Float64MultiArray()
           
            left_hand_msg.data = left_qpos.tolist()
            self.left_hand_publisher_.publish(left_hand_msg)

if __name__ == '__main__':
    rclpy.init()
    node = CommunicationNode()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Replay data from a zarr file.')
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to the zarr file')
    args = parser.parse_args()

    # 打开 zarr 文件
    loaded = zarr.open(args.file, mode='r')

    # 读取 actions
    actions_timestamps = loaded['actions/timestamps'][:]
    if actions_timestamps.size > 0:
        actions_data = loaded['actions/data'][:]  # shape: (n_samples, 26)
        actions = [(ts, data) for ts, data in zip(actions_timestamps, actions_data) if not np.isnan(ts) and not np.all(data <-200.0)]
    else:
        actions = []
    last_joint_states = None
    while rclpy.ok():
        ts,data=actions[0]
        right_arm_pose = data[:7]
        left_arm_pose = data[7:14]
        print("right_arm_pose:", right_arm_pose)    
        print("left_arm_pose:", left_arm_pose)
        # node.publish_arm_pose(right_hand_pose=right_arm_pose, left_hand_pose=left_arm_pose)
        rclpy.spin_once(node, timeout_sec=0.01)
        
        if node.latest_joint_states is not None:
            if last_joint_states is None:
                last_joint_states = node.latest_joint_states
                continue
            joint_velocities = np.abs(node.latest_joint_states - last_joint_states)
            print("joint_vel:", joint_velocities)
            if np.all(np.abs(joint_velocities) < 1e-3):
                print("All joint velocities are below 1e-3 rad/s")
                break
            else:
                print("joint vel diff:", joint_velocities)
    print("action lens:", len(actions))
    print("action ts:", actions[0][0])
    for i in range(len(actions) - 1):
        action_ts, action_data = actions[i]
        next_action_ts, _ = actions[i + 1]
        print("action ts:", action_ts)
        # 发布 hand action
        right_qpos = action_data[14:20]
        left_qpos = action_data[20:]
        right_arm_pose = action_data[:7]
        left_arm_pose = action_data[7:14]
        node.publish_hand_action(right_qpos, left_qpos)
        node.publish_arm_pose(right_hand_pose=right_arm_pose, left_hand_pose=left_arm_pose)

        # 等待下一个时间戳
        time_to_wait = (next_action_ts - action_ts) 
        time.sleep(time_to_wait)

    rclpy.shutdown()