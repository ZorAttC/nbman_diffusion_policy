import numpy as np
import json
from datetime import datetime
import sys
import os, re, subprocess
import cv2
import zarr
from threading import Lock
import threading
import queue
def keyboard_input(key_queue):
    """线程函数，用于捕捉键盘输入并将命令放入队列"""
    while True:
        key = input("Press 's' to start recording, 'q' to stop and save: ")
        if key in ['s', 'q']:
            key_queue.put(key)
        else:
            print("Invalid key. Please press 's' or 'q'.")
# ROS2
from tf2_ros import TransformBroadcaster
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, TransformStamped, WrenchStamped
from std_msgs.msg import Float64MultiArray
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.duration import Duration
def stamp_to_float64(stamp):
    return np.float64(stamp.sec) + np.float64(stamp.nanosec) * 1e-9
class Main_Node(Node):
    def __init__(self):
        super().__init__("bag_recorder")
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()
        self.lock = Lock()
        self.recording = False
        self.data = {
                'timestamps': [],#
                'states': [],#(ts,[right_arm_joint, left_arm_joint, right_hand_joint, left_hand_joint]6,6,6,6)
                'actions': [],#(ts,[right_arm_action, left_arm_action, right_hand_action, left_hand_action]7,7,6,6)
                'images': [],
                'pointclouds': []
        }

        self.latest_timeStamp=None
        # Initialize the action buffers for arm/hand data
        self.right_arm_action = np.ones(7, dtype=np.float64)*-202.0
        self.left_arm_action = np.ones(7, dtype=np.float64)*-202.0
        self.right_hand_action = np.ones(6, dtype=np.float64)*-202.0
        self.left_hand_action = np.ones(6, dtype=np.float64)*-202.0
        self.recording_freq=60.0
        # Initialize the pointcloud
        self.d435i_pointcloud = None
        # self.left_l515_pointcloud = None
        # self.right_l515_pointcloud = None

    

        self.create_timer(1.0/self.recording_freq, self.aggregate_action, callback_group=self.callback_group) 

        self.create_subscription(PointCloud2, "/camera_l515_left/depth/color/downsampled_points", self.d435i_pointcloud_callback, 10, callback_group=self.callback_group)
        self.create_subscription(Image, "/camera_l515_left/color/image_downsampled", self.d435i_image_callback, 10, callback_group=self.callback_group)
        # self.create_subscription(PointCloud2, "/camera_l515_left/depth/color/points", self.left_pointcloud_callback, 10, callback_group=self.callback_group)
        # self.create_subscription(Image, "/camera_l515_left/color/image_raw", self.lhand_image_callback, 10, callback_group=self.callback_group)
        # self.create_subscription(PointCloud2, "/camera_l515_right/depth/color/points", self.right_pointcloud_callback, 10, callback_group=self.callback_group)
        # self.create_subscription(Image, "/camera_l515_right/color/image_raw", self.lhand_image_callback, 10, callback_group=self.callback_group)
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10, callback_group=self.callback_group)

        self.create_subscription(PoseStamped, "/right_arm_ik_controller/commands", self.right_arm_callback, 10, callback_group=self.callback_group)
        self.create_subscription(PoseStamped, "/left_arm_ik_controller/commands", self.left_arm_callback, 10, callback_group=self.callback_group)
        self.create_subscription(Float64MultiArray, "/right_hand_controller/commands", self.right_hand_callback, 10, callback_group=self.callback_group)
        self.create_subscription(Float64MultiArray, "/left_hand_controller/commands", self.left_hand_callback, 10, callback_group=self.callback_group)


    def d435i_pointcloud_callback(self, msg):
        """
        解析 PointCloud2 消息，将其转换为 (n, 6) 的 NumPy 数组，格式为 [x, y, z, r, g, b]
        """
        if not self.recording:
            return

        try:
            # 从 PointCloud2 消息中提取点云数据
            points = point_cloud2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)

            # 将点云数据转换为 NumPy 数组
            point_list = []
            for point in points:
                x, y, z, rgb = point
                # 将 rgb 转换为 r, g, b
                rgb_int = int(rgb)
                r = (rgb_int >> 16) & 0xFF
                g = (rgb_int >> 8) & 0xFF
                b = rgb_int & 0xFF
                point_list.append([x, y, z, r, g, b])

            # 转换为 (n, 6) 的 NumPy 数组
            point_cloud_array = np.array(point_list, dtype=np.float32)
            self.d435i_pointcloud=((stamp_to_float64(msg.header.stamp), point_cloud_array))
            with self.lock:
                self.data['pointclouds'].append((stamp_to_float64(msg.header.stamp), point_cloud_array))
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")

    def aggregate_action(self):
        if not self.recording:
            return
        with self.lock:
            self.data['actions'].append((self.latest_timeStamp, np.hstack((self.right_arm_action, self.left_arm_action, self.right_hand_action, self.left_hand_action))))
    def joint_state_callback(self, msg):
        
        if not self.recording:
            return
      
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
           
        with self.lock:
            self.latest_timeStamp=stamp_to_float64(msg.header.stamp)
            state = np.concatenate([right_arm_joint, left_arm_joint, right_hand_joint, left_hand_joint], axis=0)
            self.data['states'].append((stamp_to_float64(msg.header.stamp), state))

    def right_arm_callback(self, msg):
        if not self.recording:
            return
        with self.lock:
            action = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 
                                  msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w],dtype=np.float64)
            
            self.right_arm_action=action

    def left_arm_callback(self, msg):
        if not self.recording:
            return
        with self.lock:
            # print("left arm callback")
            action = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 
                                  msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w],dtype=np.float64)
            self.left_arm_action=action
    def right_hand_callback(self, msg):
        if not self.recording:
            return
        with self.lock:
            self.right_hand_action=np.array(msg.data,dtype=np.float64)  #[right_pinky, right_ring, right_middle, right_index, right_thumb_motor, right_thumb_string]
    def left_hand_callback(self, msg):
        if not self.recording:
            return
        with self.lock:
            self.left_hand_action=np.array(msg.data,dtype=np.float64)  #[left_pinky, left_ring, left_middle, left_index, left_thumb_motor, left_thumb_string]
    def d435i_image_callback(self, msg):
        if not self.recording:
            return
        with self.lock:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                self.data['images'].append((stamp_to_float64(msg.header.stamp), cv_image))
            except Exception as e:
                self.get_logger().error(f"Error processing d435i image: {e}")

   

    def start_recording(self):
        self.recording = True
        self.get_logger().info("Started recording")

    def stop_recording(self, save_name):
        self.recording = False
        self.get_logger().info("Stopped recording")
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        save_path = os.path.join(data_dir, save_name)
        self.save_data(save_path)

    def save_data(self, save_path):
        with self.lock:
            # 创建 zarr 文件
            store = zarr.DirectoryStore(save_path)
            root = zarr.group(store=store, overwrite=True)
            # 存储 states
            root.create_group('states')
            if self.data['states']:
                states_timestamps = np.array([s[0] for s in self.data['states']])
                states_data = np.array([s[1] for s in self.data['states']])  # shape: (n_samples, 24)
                root['states'].array('timestamps', states_timestamps, chunks=(1000,), dtype='float64')
                root['states'].array('data', states_data, chunks=(1000, 24), dtype='float64')
                self.get_logger().info(f"Saved {len(self.data['states'])} states")
            else:
                root['states'].array('timestamps', np.array([], dtype=np.float64), chunks=(1000,), dtype='float64')
                root['states'].array('data', np.array([], dtype=np.float64).reshape(0, 24), chunks=(1000, 24), dtype='float64')
                self.get_logger().info("No states to save, storing empty arrays")

            # 存储 actions
            root.create_group('actions')
            if self.data['actions']:
                actions_timestamps = np.array([a[0] for a in self.data['actions']])
                actions_data = np.array([a[1] for a in self.data['actions']])  # shape: (n_samples, 26)
                root['actions'].array('timestamps', actions_timestamps, chunks=(1000,), dtype='float64')
                root['actions'].array('data', actions_data, chunks=(1000, 26), dtype='float64')
                self.get_logger().info(f"Saved {len(self.data['actions'])} actions")
            else:
                root['actions'].array('timestamps', np.array([], dtype=np.float64), chunks=(1000,), dtype='float64')
                root['actions'].array('data', np.array([], dtype=np.float64).reshape(0, 26), chunks=(1000, 26), dtype='float64')
                self.get_logger().info("No actions to save, storing empty arrays")

            # 存储 images (uint8)
            root.create_group('images')
            if self.data['images']:
                images_timestamps = np.array([img[0] for img in self.data['images']])
                images_data = np.array([img[1] for img in self.data['images']], dtype=np.uint8)  # shape: (n_samples, 224, 224, 3)
                root['images'].array('timestamps', images_timestamps, chunks=(1000,), dtype='float64')
                root['images'].array('data', images_data, chunks=(100, 224, 224, 3), dtype='uint8')
                self.get_logger().info(f"Saved {len(self.data['images'])} images")
            else:
                root['images'].array('timestamps', np.array([], dtype=np.float64), chunks=(1000,), dtype='float64')
                root['images'].array('data', np.array([], dtype=np.uint8).reshape(0, 224, 224, 3), chunks=(100, 224, 224, 3), dtype='uint8')
                self.get_logger().info("No images to save, storing empty arrays")

            # 存储 pointclouds (float32，每帧点数可变)
            pointclouds_group = root.create_group('pointclouds')
            if self.data['pointclouds']:
                pointclouds_timestamps = np.array([pc[0] for pc in self.data['pointclouds']])
                pointclouds_group.array('timestamps', pointclouds_timestamps, chunks=(1000,), dtype='float64')
                for i, (ts, pc) in enumerate(self.data['pointclouds']):
                    pointclouds_group.array(f'data_{i}', pc, chunks=(None, 3), dtype='float32')
                self.get_logger().info(f"Saved {len(self.data['pointclouds'])} pointclouds")
            else:
                pointclouds_group.array('timestamps', np.array([], dtype=np.float64), chunks=(1000,), dtype='float64')
                self.get_logger().info("No pointclouds to save, storing empty timestamps only")

            # 日志记录
            self.get_logger().info(f"Data saved to {save_path}")

            # 重置 self.data
            self.data = {
                'timestamps': [],
                'states': [],
                'actions': [],
                'images': [],
                'pointclouds': []
            }
def main(args=None):
    rclpy.init(args=args)
    node = Main_Node()
    executor = SingleThreadedExecutor()
    executor.add_node(node)


    # 创建队列用于线程间通信
    key_queue = queue.Queue()

    # 启动键盘输入线程
    input_thread = threading.Thread(target=keyboard_input, args=(key_queue,), daemon=True)
    input_thread.start()
    try:
        while rclpy.ok():
            # 在主线程中执行 spin 处理 ROS 回调
            executor.spin_once(timeout_sec=0.0)

            # 检查队列中是否有键盘输入
            try:
                key = key_queue.get_nowait()  # 非阻塞获取
                if key == 's':
                    node.start_recording()
                elif key == 'q':
                    save_name = 'test'+datetime.now().strftime("%Y%m%d_%H%M%S") + ".zarr"
                    node.stop_recording(save_name)
                    exit()
            except queue.Empty:
                pass  # 队列为空，继续循环

    except KeyboardInterrupt:
        print("Received Ctrl+C, shutting down...")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()