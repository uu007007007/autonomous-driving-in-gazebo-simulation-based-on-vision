import cv2
import torch
import random
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from cv_bridge import CvBridge
import sys
print(sys.path)

from ultralytics import YOLO
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose, Detection2DArray
from std_srvs.srv import SetBool
import time  # Time module for FPS calculation


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class Yolov8Node(Node):

    def __init__(self) -> None:
        super().__init__("yolo_test_node")

        self.cv_bridge = CvBridge()
        
        self.yolo = YOLO('/home/uu007007007/project_ws/src/self_driving_car_pkg/best.pt')
        
        # Topics
        
        self._sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_cb, qos_profile_sensor_data
        )



    def image_cb(self, msg: Image) -> None:
        
        # Convert image + predict
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
        cv_image=cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Run inference on the source
        results = self.yolo(cv_image)  # list of Results objects
        plots = results[0].plot()
        cv2.imshow("plot", plots)
        cv2.waitKey(1)  # Allow the window to update

def main():
    rclpy.init()
    node = Yolov8Node()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()