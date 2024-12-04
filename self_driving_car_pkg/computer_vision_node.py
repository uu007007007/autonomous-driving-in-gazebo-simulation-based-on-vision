#!/usr/bin/env python3

import cv2
import rclpy
import numpy as np
import os
import sys

from geometry_msgs.msg import Twist
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2DArray
from my_msgs.msg import AvoidMsg

from .Drive_Bot import Car, Debugging

# print문 buffer 비활성화 -> print문 바로 출력
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

# Params
STOP_DIS = 7.0  # 차량과 사람 사이의 안전 거리(m)

class Video_feed_in(Node):
    def __init__(self):

        super().__init__('video_subscriber')
        # Subscriber
        self.subscriber = self.create_subscription(Image,'/camera/image_raw',self.process_data,10)
        self.stopline_sub = self.create_subscription(Bool,'/stop_line_detected',self.stop_cb,10)
        self.yolo_sub = self.create_subscription(Detection2DArray,'/detections',self.yolo_cb,10)
        self.object_sub = self.create_subscription(AvoidMsg,'/avoid_control',self.object_avoid,10)

        #Publisher
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 40)
        timer_period = 0.03;self.timer = self.create_timer(timer_period, self.send_cmd_vel)

        # 인스턴스 변수
        self.velocity = Twist()
        self.bridge   = CvBridge() # converting ros images to opencv data
        self.Debug    = Debugging()
        self.Car      = Car()
        self.stopline_flag = False
        self.stopline_activated = False
        self.stopline_iteration = 0
        self.human_trigger = False
        self.human_stop_activated = False
        self.human_distance = None
        self.avoid = AvoidMsg()

        

    def send_cmd_vel(self):
        self.publisher.publish(self.velocity)


    def process_data(self, data):
        
        """Processes the data stream from the sensor (camera) and passes on to the
        Self Drive Algorithm which computes and executes the appropriate control
        (Steering and speed) commands.

        Args:
            data (img_msg): image data from the camera received as a ros message
        """
        self.human_str = "No Human"
        self.stop_str = "No Stop Line"
        self.obstacle_str = "No Obstacles"
        self.wait_str = None

        self.Debug.setDebugParameters()

        frame = self.bridge.imgmsg_to_cv2(data,'bgr8') # performing conversion

        Angle,Speed,img = self.Car.driveCar(frame)

        # 정지선이 인식된 경우
        if self.stopline_flag == True and self.stopline_activated == False:
            if self.stopline_iteration < 200: # 200 iteration 동안 정지
                self.stop_str = f"Stopline Detected"
                self.wait_str = f'Wait..({self.stopline_iteration}/200)'
                self.velocity.angular.z = 0.0
                self.velocity.linear.x = 0.0

            elif self.stopline_iteration < 550: # 550 iter 까지 직신
                self.stop_str = "Cross Stopline"
                self.velocity.angular.z, self.velocity.linear.x = self.angle_speed_mapping(0.0, 60)
                
            else:
                # 정지선 종료
                self.stop_str = "Stopline Finished"
                self.stopline_activated = True
                self.stopline_iteration = 0
            self.stopline_iteration += 1

        else: # 정지선이 인식되지 않은 경우에는 차선 인식 주행
            self.velocity.angular.z = Angle
            self.velocity.linear.x = Speed

        # 장애물 회피가 활성화되면 장애물 회피 주행 실시
        if self.avoid.activate:
            self.obstacle_str = "Avoiding Obstacles.."
            self.velocity.angular.z, self.velocity.linear.x = self.angle_speed_mapping(self.avoid.angle, self.avoid.speed)

        if self.human_trigger: # 사람이 근접하면 정지
            self.human_stop_activated = True
            self.human_cnt = 0
            
        if self.human_stop_activated: # 사람이 멀어지고 일정시간 이후에 다시 출발
            self.human_str = f'Human Approch ({self.human_distance:.1f}m)'
            self.wait_str = f'Wait..({self.human_cnt}/30)'
            self.velocity.angular.z = 0.0
            self.velocity.linear.x = 0.0
            self.human_cnt += 1
            if self.human_cnt == 30:
                self.human_stop_activated = False
                self.human_cnt = 0
        
        
        angle, speed = self.angle_speed_mapping_rev(self.velocity.angular.z, self.velocity.linear.x) # 각도, 속도 다시 재 맵핑
        state_img = self.state_image(angle, speed) # 상태표시 이미지 생성

        # 이미지 가로로 병합
        debug_img = cv2.hconcat( [img, state_img] )

        cv2.imshow("State", debug_img)
        cv2.waitKey(1)


    def stop_cb(self, data):
        '''
        정지선 인식 콜백 함수
        '''
        if data.data:
            self.stopline_flag = True
        

    def yolo_cb(self, bboxes):
        human_min_distance = float("inf")
        for bbox in bboxes.detections: 
            if bbox.results[0].id == "human":
                # 사람과의 거리 계산
                self.human_distance = self.calc_human_distance(bbox)
                
                if self.human_distance < human_min_distance:
                    human_min_distance = self.human_distance
        # 멈춤 또는 이동 행동 결정
        if human_min_distance <= STOP_DIS:
            self.human_trigger = True
        else:
            self.human_trigger = False
    
    
    def state_image(self, angle, speed, img_width=520, img_height=240,
                    line_gap=40, start_gap=150, x_offset = 25, y_offset = 50,
                    wheel_size=100):
              
        control_state = f'[Speed] {speed}km/h  |  [Angle] {angle:.2f}deg'
        obstacle_state = f'[Obstacle] {self.obstacle_str}'
        human_state = f'[Human] {self.human_str}'
        stopline_state = f'[Stop Line] {self.stop_str}'
        # 이미지 생성 (검은 배경)
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # 이미지 불러오기
        image = cv2.imread('/home/uu007007007/project_ws/src/self_driving_car_pkg/self_driving_car_pkg/data/steering-wheel.png')
        image = cv2.resize(image, (wheel_size, wheel_size)) # 이미지 리사이즈
        
        # 정지상태 이미지 색상 변경
        if speed == 0.0:
            # 원본 이미지에서 붉은 색 강조 (파랑과 초록 제거)
            image[:, :, 0] = 0  # 파란색 채널 제거
            image[:, :, 1] = 0  # 초록색 채널 제거
            cv2.putText(img, "STOP!", (x_offset+5, 200), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
        
        # 대기 상태 시각화
        if self.wait_str is not None:
            cv2.putText(img, self.wait_str, (start_gap, line_gap*5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # 이미지의 크기 (높이, 너비)
        h, w = image.shape[:2]

        # 하단 중심의 좌표 (너비의 중간, 높이)
        center = (w // 2, h // 2)

        # 회전 행렬 계산 (하단 중심 기준)
        rot_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # 회전된 이미지 계산
        rotated_image = cv2.warpAffine(image, rot_matrix, (w, h))

        img[y_offset:y_offset+h, x_offset:x_offset+w] = rotated_image

        cv2.putText(img, control_state, (start_gap, line_gap), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, obstacle_state, (start_gap, line_gap*2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, human_state, (start_gap, line_gap*3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, stopline_state, (start_gap, line_gap*4), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return img


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
    

    def object_avoid(self, msg):
        self.avoid.activate = msg.activate
        self.avoid.speed = msg.speed
        self.avoid.angle = msg.angle


    def angle_speed_mapping(self, angle, speed):
        angle = np.interp(angle, [-60, 60], [0.8, -0.8])
        if (speed != 0):
            speed = np.interp(speed, [30, 90], [1, 2])
        speed = float(speed)

        return angle, speed

    def angle_speed_mapping_rev(self, angle, speed):
        angle = np.interp(angle, [-0.8, 0.8], [60, -60])
        if (speed != 0):
            speed = np.interp(speed, [1, 2], [30, 90])
        speed = float(speed)

        return angle, speed

def main(args=None):
  rclpy.init(args=args)
  image_subscriber = Video_feed_in()
  rclpy.spin(image_subscriber)
  rclpy.shutdown()

if __name__ == '__main__':
	main()