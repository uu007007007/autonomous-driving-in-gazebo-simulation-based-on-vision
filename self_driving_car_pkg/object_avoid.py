import random
import math
from time import time
import os
import sys

import numpy as np
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from my_msgs.msg import Objects, AvoidMsg

# print문 buffer 비활성화 -> print문 바로 출력
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

"""
차량 파라미터
"""
CAR_WIDTH = 2.5 # 차축 길이 m
CAR_LENGTH = 3.2 # 라이다와 뒷 차축까지의 거리 m
LFD = 4.5 # 전방 주시 거리
MAX_ANGLE = 40 # 최대 조향각
AVOID_RADIUS = 3 # 회피 경로 생성 반경
DANGER_WIDTH = 3 # 위험 구역 너비
DANGER_HEIGHT = 4 # 위험 구역 길이
ANGLE_RATIO = 2 # 조향각 계수(민감도)
RAD_RATIO = AVOID_RADIUS / DANGER_HEIGHT # 거리 비율 -> 회피 경로 생성할 때 사용
MIN_RADIUS = 2.5


class ObjectAvoid(Node):
    def __init__(self):
        super().__init__('object_avoid')
        self.subscription = self.create_subscription(
            Objects,
            '/tracked_objects',
            self.tracker_callback,
            10
        )
        self.img_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(AvoidMsg, '/avoid_control', 10)
        self.image_pub = self.create_publisher(Image, "lidar_image", 10)
        self.lane_image_pub = self.create_publisher(Image, "lane_image", 10)
        timer_period = 0.03;self.timer = self.create_timer(timer_period, self.process_func)

        # 인스턴스 변수
        self.tracked_objects = []
        self.tracking_class = None
        self.tracking_id = None
        self.lfd_point = None
        self.lfd = LFD
        self.avoid_activate = False
        self.dead_reckoning = False
        self.detected_distance = None
        self.avoid_angle = 0
        self.lane_changed = False # 차선 변경 여부
        self.prev_theta_start = None

        self.roi_img = None
        self.image = None
        self.bridge = CvBridge()
        


    def tracker_callback(self, msg):
        """
        트랙커 토픽 콜백 함수 - Tracking 객체 데이터 전처리

        Args:
            msg (Objects): 객체의 좌표와 id를 담은 메세지
        """
        self.tracked_objects = []
        for obj in msg.objects:
            point_x = obj.x
            point_y = obj.y
            id = obj.id
            self.tracked_objects.append([point_x, point_y, id])

    
    def image_callback(self, msg):
        """
        이미지 콜백함수 - 이미지 전처리 후 roi 이미지 저장
        """
        # ROS Image 메시지를 OpenCV 형식으로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.image_preprocess(frame)

    
    def image_preprocess(self, frame, top_left = (490, 400), bottom_left = (360,525)):
        """
        이미지 메세지를 입력 받아 전처리

        Args:
            frame (img): 원본 이미지
            top_left (tuple): 탑뷰 전환에 사용될 이미지 좌상단 좌표 
            bottom_left (tuple): 탑뷰 전환에 사용될 이미지 좌하단 좌표
        """
        height, width, _ = frame.shape
        
        top_right = (width - top_left[0], top_left[1])
        bottom_right = (width - bottom_left[0], bottom_left[1])
        src = np.float32([top_left, top_right, bottom_right, bottom_left])
        dst = np.float32([(0, 0), (width, 0), (width, height), (0,height)])

        mat = cv2.getPerspectiveTransform(src, dst)
        trans_img = cv2.warpPerspective(frame, mat, (width, height))
        trans_img = cv2.resize(trans_img, (640, 360))
        roi_img = trans_img[trans_img.shape[0] // 2:, :]

        # HSV로 색 추출
        hsvLower = np.array([25, 51, 181])    # 추출할 색의 하한(HSV) (색상, 채도, 명도)
        hsvUpper = np.array([46, 255, 255])    # 추출할 색의 상한(HSV)
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV) # 이미지를 HSV으로 변환
        hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSV에서 마스크를 작성
        self.roi_img = cv2.bitwise_and(roi_img, roi_img, mask=hsv_mask) # 원래 이미지와 마스크를 합성



    
    def process_func(self):
        """
        실시간으로 근접한 장애물 객체에 대해 회피 경로를 생성하고, 이를 추종하기 위한 각도를 publish
        """
        start = time()
        # 경로 생성 및 추종
        self.path_planning()
        
        # 각도 publish
        self.control_pub(self.avoid_angle)

        # 시각화
        self.visualization(self.tracked_objects)
        end = time()
        print(f'연산 시간: {end-start}s')

    def path_planning(self):
        """
        위험 구역에 객체가 인식되면 그 객체를 추적하고, 회피 경로를 생성

        Results:
            self.avoid_angle (float): 회피주행을 위한 조향각

        Notes:
            - self.tracking_id 값을 이용해 근접한 객체를 추적하고, 회피 주행
        """
        detected_id_lst = [] # 감지된 id 리스트

        # 위험 구역에 들어온 객체 id 트랙킹
        for obj in self.tracked_objects:
            x, y, id = obj
            detected_id_lst.append(id)

            if self.check_danger(x, y): # 객체가 위험 구역에 진입했는지 확인
                if self.tracking_id is None: # 새로운 객체가 인식된 경우
                    # print(f'새로운 객체 진입!! 거리: {y}')
                    self.detected_distance = y
                    self.tracking_id = id # 근접한 객체의 id 저장
                    self.tracking_class = obj # 근접한 객체 정보 저장
                    self.dead_reckoning = False # 진행중이던 데드 레커닝 종료

            if self.tracking_id is not None: # 객체 트랙킹
                if id == self.tracking_id:
                    self.tracking_class = obj

        # 트래킹이 시작된 경우 회피 주행 활성화
        if self.tracking_id is not None:
            self.avoid_activate = True

        if self.tracking_id is None and self.avoid_activate: # 트래킹 종료 후 나머지 회피 주행
            # 회피 종료 탐지
            if self.detect_finish():
                self.lfd_point = None # 회피가 완료되면 초기화
                self.avoid_activate = False # 차선 복귀가 완료되었다고 판단되면 회피 주행 비활성화
                self.tracking_class = None # 객체 트랙킹 종료
                self.dead_reckoning = False # 데드 레커닝 종료
                self.detected_distance = None
                self.lane_changed = False # 차선 변경 초기화
                self.avoid_angle = 0 # 회피 각도 초기화

        # 회피 주행이 활성화된 경우
        if self.avoid_activate:

            if self.dead_reckoning:
                self.tracking_class[1] -= 0.01 # 데드 레커닝이 수행되는 동안 객체의 y축 위치를 0.1m씩 감소시킴

            tracking_x, tracking_y, _ = self.tracking_class
            x_points, y_points = self.create_path(tracking_x, tracking_y) # 회피 경로 생성
            self.purepursuit(x_points, y_points) # 경로 추종 각도 할당

            # 트래킹 하던 객체가 사라지면 트래킹 종료
            if self.tracking_id not in detected_id_lst:
                self.tracking_id = None
                self.dead_reckoning = True # 데드 레커닝 시작

        print(f'차선 변경 여부: {self.lane_changed}')

        # 이미지 Pub
        if self.image is not None and self.roi_img is not None:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.image, encoding="bgr8"))
            self.lane_image_pub.publish(self.bridge.cv2_to_imgmsg(self.roi_img, encoding="bgr8"))
            # cv2.imshow("avoid", lidar_img)

        # print(f'회피주행 활성화 여부: {self.avoid_activate}')
    
    
    def check_danger(self, x, y):
        """
        객체가 위험 구역에 진입했는지 확인

        Args:
            x (float): 인식된 객체의 x좌표
            y (float): 인식된 객체의 y좌표

        Returs:
            True/False (Bool): 영역내에 근접 여부
        """
        return ((x < DANGER_WIDTH/2) and (x > -DANGER_WIDTH/2) and (y < DANGER_HEIGHT)) # 범위내에 진입하면 True를 반환
    
    
    def create_path(self, x, y):
        """
        트래킹 된 객체의 위치를 기반으로 객체의 왼쪽 반원호 형태의 회피 경로 생성

        Args:
            x (float): 객체의 x좌표
            y (float): 객체의 y좌표

        Returns:
            x_points (list of float): 회피 경로 포인트의 x좌표 리스트
            y_points (list of float): 회피 경로 포인트의 y좌표 리스트
        
        """
        # 중심 좌표와 반지름
        center_x = x
        center_y = y
        radius = max(RAD_RATIO * self.detected_distance, MIN_RADIUS) # 객체 인식 거리에 계수를 곱하여 반경 설정


        # 각도 범위: 90도에서 270도까지 (단위: 라디안)
        if self.lane_changed and self.dead_reckoning == False: # 차선이 변경된 경우에는 경로를 반대 방향으로 생성
            theta_start = np.radians(-90)
        elif self.lane_changed == False and self.dead_reckoning == False:
            theta_start = np.radians(270)   # 270도 -> 라디안
        else:
            theta_start = self.prev_theta_start # 데드 레커닝이 진행중일 때는 이전 방향 유지
        
        self.prev_theta_start = theta_start # 이전 방향 저장
        theta_end = np.radians(90)  # 90도 -> 라디안
        

        # 각도를 일정 간격으로 나누어 점 생성
        num_points = 100  # 생성할 점의 수
        theta_values = np.linspace(theta_start, theta_end, num_points)

        # 점들의 좌표 계산
        x_points = center_x + radius * np.cos(theta_values)
        y_points = center_y + radius * np.sin(theta_values)
        
        return x_points, y_points


    def purepursuit(self, x_points, y_points):
        """
        Pure Pursuit 알고리즘 연산을 수행하여 경로를 추종하기 위한 angle을 도출합니다.

        Args:
            x_points (list of float): x 좌표값들의 리스트
            y_points (list of float): y 좌표값들의 리스트

        Results:
            self.avoid_angle (float): 계산된 조향각

        Notes:
            - look forward distance에 해당하는 포인트 좌표를 인스턴스 변수로 저장하여 visualization에 활용
        """
        is_look_forward_point = False
        self.lfd_point = None

        for x, y in zip(x_points, y_points): #path 포인트 순회
            dx = x
            dy = y + CAR_LENGTH
            dis = math.sqrt(pow(dx, 2)+pow(dy, 2)) # 차량의 후방에서부터 path 까지의 거리
            
            if dy > 0: #차량보다 앞에 있는 점들만 순회
                if dis>= self.lfd : #dis 가 lfd 보다 큰 경우 break
                    is_look_forward_point=True #lfd 활성화
                    self.lfd_point = (x,y)
                    break

        if is_look_forward_point: #lfd 가 활성화 되었을때
            steering = math.atan((2 * CAR_LENGTH * (dx / dis)) / dis) # PurePursuit 알고리즘 계산
            angle = (math.degrees(steering)) #조향각 degree로 전환

            angle = angle * ANGLE_RATIO

            # 각도 제한
            if angle >= MAX_ANGLE:
                angle = MAX_ANGLE
            elif angle <= -MAX_ANGLE:
                angle = -MAX_ANGLE

            self.avoid_angle = angle


    def detect_finish(self, width_threshold=0.8):
        """
        회피 주행 도중 회피 주행 종료시점은 인식하는 함수

        - roi 이미지에서 외곽선을 검출해 조건에 부합하면 True를 반환

        Returns:
            True/False (bool): 종료 조건에 부합한지 여부
        """
        finish_flag = False
        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(self.roi_img, cv2.COLOR_BGR2GRAY)
        # 블러 적용하여 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 엣지 검출
        edges = cv2.Canny(blurred, 50, 150)
        #직선 검출
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, None, 100, 50)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(self.roi_img, (x1,y1), (x2, y2), (0,0,255), 2)
               
                # 직선의 기울기 계산
                if (x2 - x1) == 0:
                    continue # 0으로 나누는 경우 무시
                else:
                    gradient = (y2 - y1)/(x2 - x1)
                
                # 차선 변경 여부 할당
                self.lane_changed = (gradient < 0 and x1 > (self.roi_img.shape[1] * (1-width_threshold)) and
                               x1 < (self.roi_img.shape[1] / 2))

                # 종료 조건 반환
                finish_flag = (gradient > 0 and x1 < (self.roi_img.shape[1] * width_threshold) and
                               x1 > (self.roi_img.shape[1] / 2))
        
        return finish_flag
                    



    def control_pub(self, angle, speed = 40):
        """
        회피 제어 값을 publish

        Args:
            angle (float): 차량의 조향각
            speed (float): 차량의 속도(기본: 30)

        Pub:
            - self.avoid_activate (bool): 회피 활성화 여부
            - speed (float32)
            - angle (float32)
        """
        msg = AvoidMsg()
        msg.activate = self.avoid_activate
        msg.speed = float(speed)
        msg.angle = float(angle)
        self.publisher.publish(msg)


    def visualization(self, tracked_objects, car_size=100):
        """
        OpenCV를 이용해 객체의 상태와 경로 및 lfd 시각화

        Args:
            tracked_objects (list of tuple): 트랙킹되고 있는 객체들의 정보를 담은 리스트
        """
        # 차량 이미지 불러오기
        car_img = cv2.imread('/home/uu007007007/project_ws/src/self_driving_car_pkg/self_driving_car_pkg/data/car.png')
        car_img = cv2.resize(car_img, (car_size, car_size)) # 이미지 리사이즈

        # 이미지 크기 (픽셀 단위) - 크기 줄이기
        img_width = 640  # 이미지 너비
        img_height = 360  # 이미지 높이

        # 이미지 배율
        meter_to_pixel = 20  # 미터당 40픽셀로 설정 (배율 줄이기)

        # 라이다 위치 설정 (이미지 내 좌표)
        vehicle_length = 3 * meter_to_pixel  # 차량 길이 (픽셀 단위)
        lidar_position = (int(img_width / 2), int(img_height - vehicle_length))  # 차량의 맨 하단 중앙

        # 라이다 감지 범위
        lidar_range = 12  # 라이다 반경 (미터)
       
        lidar_radius_pixels = lidar_range * meter_to_pixel  # 라이다 반경 (픽셀 단위)
        lidar_angle = 180  # 라이다 감지 각도 (도)

        # 이미지 생성 (검은 배경)
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # 라이다 반경을 반투명 반원으로 그리기
        overlay = img.copy()  # 원본 이미지 복사
        cv2.ellipse(overlay, lidar_position, (lidar_radius_pixels, lidar_radius_pixels), 0, lidar_angle, 360, (20, 100, 20), -1)

        # 위험 구역 사각형으로 표시
        danger_zone_width = DANGER_WIDTH * meter_to_pixel # 위험 구역 크기 (단위: 픽셀)
        danger_zone_height = DANGER_HEIGHT * meter_to_pixel
        danger_zone_top_left = (int(lidar_position[0] - danger_zone_width//2), int(lidar_position[1] - danger_zone_height))
        danger_zone_bottom_right = (int(lidar_position[0] + danger_zone_width//2), int(lidar_position[1]))
        cv2.rectangle(overlay, danger_zone_top_left, danger_zone_bottom_right, (0, 0, 150), -1)

        # 반투명 효과
        alpha = 0.6 # 투명도
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)  # 반투명 효과 (alpha는 투명도)

        # 차량 이미지 삽입
        car_offset_x = int(lidar_position[0] - car_size // 2)
        car_offset_y = int(lidar_position[1])
        
        # car_img 크기와 삽입 범위가 img 크기를 넘지 않도록 확인
        end_x = car_offset_x + car_size
        end_y = car_offset_y + car_size

        # img의 크기 제한에 맞게 end_x, end_y 조정
        if end_x > img.shape[1]:
            end_x = img.shape[1]
        if end_y > img.shape[0]:
            end_y = img.shape[0]

        # 삽입 범위가 유효한지 확인하고 car_img 크기를 맞춰서 삽입
        img[car_offset_y:end_y, car_offset_x:end_x] = car_img[:end_y - car_offset_y, :end_x - car_offset_x]

        # 라이다 위치에 점 찍기
        cv2.circle(img, lidar_position, 5, (0, 200, 255), -1)

        #lfd 반경 시각화
        cv2.circle(img, (lidar_position[0], int(lidar_position[1] + (CAR_LENGTH * meter_to_pixel))),
                   int(self.lfd * meter_to_pixel), (200, 200, 255), 1)
        
        # 트래킹 된 객체에 대해 경로 표시
        if self.avoid_activate:
            x = int(lidar_position[0] + self.tracking_class[0] * meter_to_pixel)
            y = int(lidar_position[1] - self.tracking_class[1] * meter_to_pixel)
            x_points, y_points = self.create_path_visualization(x, y, meter_to_pixel) # 시각화를 위한 회피 경로 생성
            # 각 점을 이미지에 표시
            for x, y in zip(x_points, y_points):
                # 점의 좌표를 정수로 변환
                cv2.circle(img, (int(x), int(y)), 1, (255, 255, 0), -1)  # 하늘색 점

        # lfd 포인트 시각화
        if self.lfd_point is not None:
            lfd_x = int(lidar_position[0] + self.lfd_point[0] * meter_to_pixel)
            lfd_y = int(lidar_position[1] - self.lfd_point[1] * meter_to_pixel)
            cv2.circle(img, (lfd_x, lfd_y), 3, (0, 0, 255), -1)  # 빨간색 점

        # 탐지된 객체 좌표 표시
        if len(tracked_objects) != 0:
            for obj in tracked_objects:
                # 객체의 중심점 (obj[0], obj[1]은 미터 단위로 가정)
                # 객체 위치를 픽셀로 변환
                obj_x = int(lidar_position[0] + obj[0] * meter_to_pixel)
                obj_y = int(lidar_position[1] - obj[1] * meter_to_pixel)
                color = self.get_class_color(obj[2])  # 클래스에 맞는 색상 생성
                # 객체 중심점에 원 그리기
                cv2.circle(img, (obj_x, obj_y), 10, color, -1)

                # 객체의 거리 (미터 단위)
                distance = np.sqrt(obj[0]**2 + obj[1]**2)  # (x, y)로부터 원점(0,0)까지의 거리 계산
                distance_text = f"Dist: {distance:.2f}m"
                class_text = f"Id: {obj[2]}"  # 클래스 번호를 텍스트로 표시

                # 텍스트 추가 (거리와 클래스 번호)
                cv2.putText(img, distance_text, (obj_x + 15, obj_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_8)
                cv2.putText(img, class_text, (obj_x + 15, obj_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_8)

        self.image = img

        # # 화면에 이미지 표시
        # cv2.imshow('Lidar Visualization', img)
        # cv2.waitKey(1)


    def get_class_color(self, class_id):
        """
        고유한 색상을 생성하기 위해 hash를 사용하여 일정한 색상을 생성
        """
        random.seed(class_id*12654)  # 클래스 ID에 따라 시드를 고정하여 동일한 색상을 부여
        return (random.randint(130, 255), random.randint(100, 255), random.randint(100, 255))


    def create_path_visualization(self, x, y, meter_to_pixel):
        """
        회피 경로 시각화를 위해 cv에 활용할 수 있는 좌표로 계산
        """
        # 중심 좌표와 반지름
        center_x = x
        center_y = y
        radius = max(RAD_RATIO * self.detected_distance, MIN_RADIUS) * meter_to_pixel

        print(f'회피 활성화 {self.avoid_activate}, 차선변경 {self.lane_changed}')
        # 각도 범위: 90도에서 270도까지 (단위: 라디안)
        if self.lane_changed and self.dead_reckoning == False: # 차선이 변경된 경우에는 경로를 반대 방향으로 생성
            theta_start = np.radians(-90)
        elif self.lane_changed == False and self.dead_reckoning == False:
            theta_start = np.radians(270)   # 270도 -> 라디안
        else:
            theta_start = self.prev_theta_start # 데드 레커닝이 진행중일 때는 이전 방향 유지
        
        self.prev_theta_start = theta_start # 이전 방향 저장
        theta_end = np.radians(90)  # 90도 -> 라디안

        # 각도를 일정 간격으로 나누어 점 생성
        num_points = 100  # 생성할 점의 수
        theta_values = np.linspace(theta_start, theta_end, num_points)

        # 점들의 좌표 계산
        x_points = center_x + radius * np.cos(theta_values)
        y_points = center_y + radius * np.sin(theta_values)

        return x_points, y_points
    

    


def main(args=None):
    
    rclpy.init(args=args)
    tracker = ObjectAvoid()
    rclpy.spin(tracker)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
