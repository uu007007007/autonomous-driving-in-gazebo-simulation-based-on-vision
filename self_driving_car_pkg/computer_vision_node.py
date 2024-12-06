#!/usr/bin/env python3

import cv2
import rclpy
import numpy as np
import os
import sys
from time import time

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
STOP_DIS = 6.0  # 차량과 사람 사이의 안전 거리(m)
HUMAN_CONF = 0.9
HUMAN_WAIT_T = 2.0
STOPLINE_WAIT_T = 7.0
STOPLINE_CROSS_T = 10.0

"""
시각화 파라미터
"""
BORDER_SIZE = 1 # 테두리 두께
BORDER_COLOR = (255, 255, 255) # 테두리 색상 (흰색)
LIDAR_TITLE = "[OBSTACLE DETECTION]"
YOLO_TITLE = "[HUMAN DETECTION]"
LANE_TITLE = "[LANE CHECK]"
STOPLINE_TITLE = "[STOP LINE DETECTION]"
STATE_TITLE = "[STATE]"




class ComputerVision(Node):
    def __init__(self):

        super().__init__('computer_vision_node')
        # Subscriber
        self.subscriber = self.create_subscription(Image,'/camera/image_raw',self.process_data,10) # 카메라 이미지 sub
        self.lidar_image_sub = self.create_subscription(Image,'/lidar_image',self.lidar_img_cb,10) # 라이다 이미지 sub
        self.lane_image_sub = self.create_subscription(Image,'/lane_image',self.lane_img_cb,10) # lane 이미지 sub
        self.yolo_image_sub = self.create_subscription(Image,'/yolo_image',self.yolo_img_cb,10) # 욜로 이미지 sub
        self.stopline_image_sub = self.create_subscription(Image,'/stopline_image',self.stopline_img_cb,10) # 라이다 이미지 sub

        self.stopline_sub = self.create_subscription(Bool,'/stop_line_detected',self.stop_cb,10) # 정지선 인식 결과 sub
        self.yolo_sub = self.create_subscription(Detection2DArray,'/detections',self.yolo_cb,10) # 욜로 인식 결과 sub 
        self.object_sub = self.create_subscription(AvoidMsg,'/avoid_control',self.object_avoid,10) # 회피 주행 결과 sub

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
        self.stopline_cnt = 0
        self.human_trigger = False
        self.human_stop_activated = False
        self.human_distance = None
        self.avoid = AvoidMsg()

        # 이미지 변수
        self.yolo_img = None
        self.lidar_img = None
        self.lane_img = None
        self.stopline_img = None
        self.cross_img = None

        

    def send_cmd_vel(self):
        self.publisher.publish(self.velocity)


    def process_data(self, data):
        
        """Processes the data stream from the sensor (camera) and passes on to the
        Self Drive Algorithm which computes and executes the appropriate control
        (Steering and speed) commands.

        Args:
            data (img_msg): image data from the camera received as a ros message
        """
        self.human_str = "None"
        self.stop_str = "None"
        self.obstacle_str = "None"
        self.wait_str = None

        self.Debug.setDebugParameters()

        frame = self.bridge.imgmsg_to_cv2(data,'bgr8') # performing conversion

        Angle,Speed,img = self.Car.driveCar(frame)

        # 정지선이 인식된 경우
        if self.stopline_flag == True and self.stopline_activated == False:
            if not self.stopline_cnt:
                self.stopline_cnt = time()
            stopline_wait = time()-self.stopline_cnt
            if stopline_wait < STOPLINE_WAIT_T: # 5초 동안 정지
                self.stop_str = f"Stopline Detected"
                self.wait_str = f'Wait..({stopline_wait:.1f}/{STOPLINE_WAIT_T}s)'
                self.velocity.angular.z = 0.0
                self.velocity.linear.x = 0.0

            elif stopline_wait < STOPLINE_WAIT_T + STOPLINE_CROSS_T: # 횡단보도 주행
                self.stop_str = "Cross Stopline"
                yaw = self.stopline_cross()
                self.velocity.angular.z, self.velocity.linear.x = self.angle_speed_mapping(yaw, 60)
                
            else:
                # 정지선 종료
                self.stopline_activated = True
                self.stopline_cnt = 0
                self.cross_img = None
            # self.stopline_cnt += 1

        else: # 정지선이 인식되지 않은 경우에는 차선 인식 주행
            self.velocity.angular.z = Angle
            self.velocity.linear.x = Speed

        # 장애물 회피가 활성화되면 장애물 회피 주행 실시
        if self.avoid.activate:
            self.obstacle_str = "Avoiding Obstacles.."
            self.velocity.angular.z, self.velocity.linear.x = self.angle_speed_mapping(self.avoid.angle, self.avoid.speed)

        if self.human_trigger: # 사람이 근접하면 정지
            self.human_stop_activated = True
            self.human_cnt = time()
            
        if self.human_stop_activated: # 사람이 멀어지고 일정시간 이후에 다시 출발
            human_wait = time()-self.human_cnt
            self.human_str = f'Human Approch ({self.human_distance:.1f}m)'
            self.wait_str = f'Wait..({human_wait:.1f}/{HUMAN_WAIT_T}s)'
            self.velocity.angular.z = 0.0
            self.velocity.linear.x = 0.0
            # self.human_cnt += 1
            if human_wait >= HUMAN_WAIT_T:
                self.human_stop_activated = False
                self.human_cnt = 0
        
        
        angle, speed = self.angle_speed_mapping_rev(self.velocity.angular.z, self.velocity.linear.x) # 각도, 속도 다시 재 맵핑
        state_img = self.state_image(angle, speed) # 상태표시 이미지 생성

        self.image_merge(img, state_img)




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
                if bbox.results[0].score > HUMAN_CONF:
                    # 사람과의 거리 계산
                    self.human_distance = self.calc_human_distance(bbox)
                    
                    if self.human_distance < human_min_distance:
                        human_min_distance = self.human_distance
        # 멈춤 또는 이동 행동 결정
        if human_min_distance <= STOP_DIS:
            self.human_trigger = True
        else:
            self.human_trigger = False


    def yolo_img_cb(self, img:Image)-> None:
        self.yolo_img = self.bridge.imgmsg_to_cv2(img,'bgr8') # performing conversion

    
    def lidar_img_cb(self, img:Image)-> None:
        self.lidar_img = self.bridge.imgmsg_to_cv2(img,'bgr8') # performing conversion


    def lane_img_cb(self, img:Image)-> None:
        self.lane_img = self.bridge.imgmsg_to_cv2(img,'bgr8') # performing conversion


    def stopline_img_cb(self, img:Image)-> None:
        self.stopline_img = self.bridge.imgmsg_to_cv2(img,'bgr8') # performing conversion
    
    
    def state_image(self, angle, speed, img_width=741, img_height=240,
                    line_gap=40, start_gap=200, wheel_size=100):
              
        control_state = f'[Speed] {speed}km/h  |  [Angle] {angle:.2f}deg'
        obstacle_state = f'[Obstacle] {self.obstacle_str}'
        human_state = f'[Human] {self.human_str}'
        stopline_state = f'[Stop Line] {self.stop_str}'
        # 이미지 생성 (검은 배경)
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # 이미지 불러오기
        image = cv2.imread('/home/uu007007007/project_ws/src/self_driving_car_pkg/self_driving_car_pkg/data/steering-wheel.png')
        image = cv2.resize(image, (wheel_size, wheel_size)) # 이미지 리사이즈
        
        x_offset = (start_gap-wheel_size)//2
        y_offset = 50

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

        # 핸들 중심의 좌표
        center = (w // 2, h // 2)

        # 회전 행렬 계산 (중심 기준)
        rot_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # 회전된 이미지 계산
        rotated_image = cv2.warpAffine(image, rot_matrix, (w, h))

        img[y_offset:y_offset+h, x_offset:x_offset+w] = rotated_image

        text_color = [(150, 150, 150), (150, 150, 150), (150, 150, 150)] # 텍스트별 색상 저장
        text_list = [self.obstacle_str, self.human_str, self.stop_str] # 텍스트 리스트

        for i, text in enumerate(text_list): # "None"이 아닌 텍스트의 색상 변경
            if text != "None":
                text_color[i] = (255, 255, 0) # 글자색 하늘색

        cv2.putText(img, control_state, (start_gap, line_gap), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, obstacle_state, (start_gap, line_gap*2), cv2.FONT_HERSHEY_COMPLEX, 0.5, text_color[0], 1, cv2.LINE_AA)
        cv2.putText(img, human_state, (start_gap, line_gap*3), cv2.FONT_HERSHEY_COMPLEX, 0.5, text_color[1], 1, cv2.LINE_AA)
        cv2.putText(img, stopline_state, (start_gap, line_gap*4), cv2.FONT_HERSHEY_COMPLEX, 0.5, text_color[2], 1, cv2.LINE_AA)

        return img
    

    def image_merge(self, lane_img, state_img, x_offset=10, y_offset=20, font_size=0.6, font_color=(0, 255, 0)):
        """
        출력할 이미지들을 모두 병합하여 출력

        Args:
            lane_img (array): 차선 인식과 표지판, 신호등 인식 정보가 담긴 이미지
            state_img (array): 차량의 조향각, 속도, 상태 정보가 담긴 이미지

        Notes:
            - 각각의 이미지에 타이틀 및 테두리 추가
            - 욜로 이미지, 라이다 이미지, 정지선 이미지, 상태이미지를 모두 병합
        """
        # 공백을 맞추기 위한 검은 배경 생성
        black_board_height = lane_img.shape[0]
        black_board_width = 221
        # 이미지 생성 (검은 배경)
        black_img = np.zeros((black_board_height, black_board_width, 3), dtype=np.uint8)

        # 차선 인식, 상태 이미지 가로로 병합
        debug_img = cv2.hconcat( [lane_img, state_img] )
        debug_img = cv2.hconcat( [black_img, debug_img] )

        # 상태 이미지 타이틀 추가
        cv2.putText(debug_img, STATE_TITLE, (x_offset, y_offset),
                    cv2.FONT_HERSHEY_COMPLEX, font_size, font_color, 1, cv2.LINE_AA)

        # 상태 이미지 테두리 추가
        state = cv2.copyMakeBorder(debug_img, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, 
                                   cv2.BORDER_CONSTANT, value=BORDER_COLOR)

        if (self.yolo_img is not None and self.lidar_img is not None and
            self.lane_img is not None and self.stopline_img is not None):

            if self.cross_img is not None:
                self.stopline_img = self.cross_img # 횡단 보도 주행 이미지 할당

            # 이미지 타이틀 추가
            lidar = cv2.putText(self.lidar_img.copy(), LIDAR_TITLE, (x_offset, y_offset),
                        cv2.FONT_HERSHEY_COMPLEX, font_size, font_color, 1, cv2.LINE_AA) # 욜로 이미지 타이틀 추가
            lane = cv2.putText(self.lane_img.copy(), LANE_TITLE, (x_offset, y_offset),
                        cv2.FONT_HERSHEY_COMPLEX, font_size, font_color, 1, cv2.LINE_AA) # 욜로 이미지 타이틀 추가
            yolo = cv2.putText(self.yolo_img.copy(), YOLO_TITLE, (x_offset, y_offset),
                        cv2.FONT_HERSHEY_COMPLEX, font_size, font_color, 1, cv2.LINE_AA) # 욜로 이미지 타이틀 추가
            stopline = cv2.putText(self.stopline_img.copy(), STOPLINE_TITLE, (x_offset, y_offset),
                        cv2.FONT_HERSHEY_COMPLEX, font_size, font_color, 1, cv2.LINE_AA) # 정지선 이미지 타이틀 추가
            
            # 이미지 테두리 추가
            lidar = cv2.copyMakeBorder(lidar, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, 
                                            cv2.BORDER_CONSTANT, value=BORDER_COLOR)
            lane = cv2.copyMakeBorder(lane, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, 
                                            cv2.BORDER_CONSTANT, value=BORDER_COLOR)
            yolo = cv2.copyMakeBorder(yolo, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, 
                                    cv2.BORDER_CONSTANT, value=BORDER_COLOR)
            stopline = cv2.copyMakeBorder(stopline, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, 
                                        cv2.BORDER_CONSTANT, value=BORDER_COLOR)
            # 라이다, lane 이미지 세로로 병합
            lidar_img = cv2.vconcat( [lidar, lane] )

            # 욜로, 정지선 이미지 세로로 병합
            yolo_stopline_img = cv2.vconcat( [yolo, stopline] )

            # 라이다, 욜로,정지선 이미지 가로로 병합
            sensor_img = cv2.hconcat( [lidar_img, yolo_stopline_img] )

            # 전체 이미지 병합
            whole_img = cv2.vconcat( [sensor_img, state] )
        
            cv2.imshow("State", whole_img)
            cv2.waitKey(1)


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


    def stopline_cross(self, term_with_line=200):
        """
        횡단보도를 건너는 동안 차선을 유지하는 조향각을 계산하는 함수

        Returns:
            angle (float): 차선을 유지하기 위한 조향각
        """
        yellow_line_img = self.lane_img.copy()
        # 이미지 사이즈
        h, w = yellow_line_img.shape[:2]
        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(yellow_line_img, cv2.COLOR_BGR2GRAY)
        # 블러 적용하여 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 엣지 검출
        edges = cv2.Canny(blurred, 50, 150)
        #직선 검출
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, None, 100, 50)

        car_center = (w // 2, h // 2)
        # 차량 중심 표시
        cv2.circle(yellow_line_img, (int(car_center[0]), int(car_center[1])), 10, (0, 0, 255), 2)

        centers_of_lines = [] # 차선 중심 리스트

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # 직선의 기울기 계산
                if (x2 - x1) == 0:
                    continue # 0으로 나누는 경우 무시
                else:
                    gradient = (y2 - y1)/(x2 - x1)

                # 오른쪽 실선 인식
                if gradient > 0:
                    cv2.line(yellow_line_img, (x1,y1), (x2, y2), (0,255,0), 3) # 초록색으로 표시

                    line_center = (x1 + x2) / 2
                    centers_of_lines.append(line_center)
        line_point = min(centers_of_lines)
        goal_center = (line_point - term_with_line, car_center[1])
        cv2.line(yellow_line_img, (int(goal_center[0]),int(goal_center[1])),
                    (int(line_point), int(car_center[1])), (255,0,0), 2) # 직선부터 중앙까지 표시
        cv2.circle(yellow_line_img, (int(goal_center[0]), int(goal_center[1])), 5, (255,0,0), -1)

        self.cross_img = yellow_line_img

        angle = (goal_center[0] - car_center[0]) / 3
        
        return angle



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
  image_subscriber = ComputerVision()
  rclpy.spin(image_subscriber)
  rclpy.shutdown()

if __name__ == '__main__':
	main()