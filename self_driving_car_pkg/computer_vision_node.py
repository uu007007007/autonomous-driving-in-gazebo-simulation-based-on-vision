#!/usr/bin/env python3

import cv2
from geometry_msgs.msg import Twist
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2DArray
import rclpy

from .Drive_Bot import Car, Debugging

class Video_feed_in(Node):
    def __init__(self):

        super().__init__('video_subscriber')
        self.subscriber = self.create_subscription(Image,'/camera/image_raw',self.process_data,10)
        self.stopline_sub = self.create_subscription(Bool,'/stop_line_detected',self.stop_cb,10)
        self.yolo_sub = self.create_subscription(Detection2DArray,'/detections',self.yolo_cb,10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 40)
        timer_period = 0.5;self.timer = self.create_timer(timer_period, self.send_cmd_vel)

        self.velocity = Twist()
        self.bridge   = CvBridge() # converting ros images to opencv data
        self.Debug    = Debugging()
        self.Car      = Car()
        self.stopline_flag = False
        self.stopline_activated = False
        self.stopline_iteration = 0
        self.human_trigger = False
        self.human_stop_activated = False

        # 파라미터
        self.stop_distance = 7.0  # 차량과 사람 사이의 안전 거리(m)

    def send_cmd_vel(self):
        self.publisher.publish(self.velocity)

    def process_data(self, data):
        
        """Processes the data stream from the sensor (camera) and passes on to the
        Self Drive Algorithm which computes and executes the appropriate control
        (Steering and speed) commands.

        Args:
            data (img_msg): image data from the camera received as a ros message
        """
        self.Debug.setDebugParameters()

        frame = self.bridge.imgmsg_to_cv2(data,'bgr8') # performing conversion

        Angle,Speed,img = self.Car.driveCar(frame)
        # 정지선이 인식된 경우
        if self.stopline_flag == True and self.stopline_activated == False:
            if self.stopline_iteration < 200: # 200 iteration 동안 정지
                print("stopline detected!!")
                self.velocity.angular.z = 0.0
                self.velocity.linear.x = 0.0
            elif self.stopline_iteration < 550:
                print("cross stopline!!")
                self.velocity.angular.z = 0.0
                self.velocity.linear.x = 1.5
            else:
                print("stop finished!!")
                self.stopline_activated = True
                self.stopline_iteration = 0
            self.stopline_iteration += 1
        else:
            self.velocity.angular.z = Angle
            self.velocity.linear.x = Speed
        if self.human_trigger:
            self.human_stop_activated = True
            self.human_cnt = 0
            
        if self.human_stop_activated:    
            print("human approch!!  count :", self.human_cnt)
            self.velocity.angular.z = 0.0
            self.velocity.linear.x = 0.0
            self.human_cnt += 1
            if self.human_cnt == 50:
                print("human disapeared!!")
                self.human_stop_activated = False
                self.human_cnt = 0


        cv2.imshow("Frame",img)
        cv2.waitKey(1)

    def stop_cb(self, data):
        '''
        정지선 인식 콜백 함수
        '''
        if data.data == True:
            self.stopline_flag = True
        
    def yolo_cb(self, bboxes):
        human_min_distance = float("inf")
        for bbox in bboxes.detections: 
            if bbox.results[0].id == "human":
                # 사람과의 거리 계산
                human_distance = self.calc_human_distance(bbox)
                self.get_logger().info(f"Human detected at {human_distance:.2f} meters.")
                if human_distance < human_min_distance:
                    human_min_distance = human_distance
        # 멈춤 또는 이동 행동 결정
        if human_min_distance <= self.stop_distance:
            self.human_trigger = True
        else:
            self.human_trigger = False
    
    def calc_human_distance(self, detection):
        """
        감지된 사람의 거리 계산.

        Args:
            detection (vision_msgs.msg.Detection2D): 감지된 객체의 바운딩 박스 정보.

        Returns:
            float: 차량과 사람 사이의 거리(m).
        """
        H_camera = 150  # 카메라의 실제 높이(cm)
        focal_length = 687  # 카메라의 초점 거리(픽셀)
        
        # y축 거리 계산 (사람의 바운딩 박스 높이)
        person_height_in_pixels = detection.bbox.size_y

        # 거리 계산
        distance_cm = (H_camera * focal_length) / person_height_in_pixels
        return distance_cm / 100  # cm를 m로 변환

def main(args=None):
  rclpy.init(args=args)
  image_subscriber = Video_feed_in()
  rclpy.spin(image_subscriber)
  rclpy.shutdown()

if __name__ == '__main__':
	main()