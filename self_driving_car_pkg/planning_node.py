import rclpy
from rclpy.node import Node
from my_msgs.msg import TrafficMsg
from std_msgs.msg import String, Bool
from vision_msgs.msg import Detection2DArray
import time

import os
import sys

# print문 buffer 비활성화 -> print문 바로 출력
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')

        # ROS 퍼블리셔 초기화
        self.action_pub = self.create_publisher(String, '/action', 10)

        # ROS 이미지 구독자 초기화
        self.traffic_sub = self.create_subscription(TrafficMsg,'/traffic_state',self.traffic_callback,10)
        self.yolo_sub = self.create_subscription(Detection2DArray,'/detections',self.yolo_callback,10)
        self.trigger_sub = self.create_subscription(Bool,'/trigger',self.trigger_callback,10)
        timer_period = 0.03;self.timer = self.create_timer(timer_period, self.planning_pub)
        
        # 신호등 관련 변수
        self.TrafficLight_iterations = 0
        self.GO_MODE_ACTIVATED = False
        self.STOP_MODE_ACTIVATED = False

        # 표지판 관련 변수
        self.closest_signal = None
        self.Mode = "Detection"
        self.prev_Mode_Turn = "Detection"
        self.direction = None

        # perception 으로부터 받는 데이터
        self.traffic_state = None
        self.close_proximity = None
        self.bounding_boxes = None

        # 트리커 작동 여부
        self.trigger = None

    # 신호등
    def traffic_callback(self, data):
        
        self.traffic_state = data.state
        self.close_proximity = data.proximity

    # 트리거
    def trigger_callback(self, data):
        self.trigger = data.data

    # 객체 인식
    def yolo_callback(self, data):
        self.bounding_boxes = data


    def planning_pub(self):
        current_state = [self.traffic_state, self.close_proximity, self.bounding_boxes, self.trigger]
        # print(current_state)

        if all(ele != None for ele in current_state): # current_state 값이 모두 할당되어야 실행
            if self.trigger == True: # 트리커가 발동하면 강제 stop 및 process 중단
                action = "stop"
                print('Trigger Activated!!!')
            else:
                action = self.process(current_state)
            # print(f'action : {action}')
            action_msg = String()
            action_msg.data = action
            self.action_pub.publish(action_msg)
            # # Log the state for debugging
            # self.get_logger().info(f"Action State: {action}")

    def process(self, current_state):
        bounding_boxes = current_state[2]
        traffic_elements = current_state[0:2]
        traffic_action = self.process_traffic_light(traffic_elements) # 신호등 action 결정
        signal_action = self.process_signal(bounding_boxes) # 표지판 action 결정

        # 우선 순위 결정
        if traffic_action != "none": # 신호를 우선으로 처리
            return traffic_action
        else:
            return signal_action
        

    def process_traffic_light(self, traffic_elements):
        Traffic_State = traffic_elements[0]
        CloseProximity = traffic_elements[1]

        action = "none" # action default 값

        if((Traffic_State == "Stop") and CloseProximity): # 빨간불 + 가까워진 경우
            action = "stop"
            self.STOP_MODE_ACTIVATED = True
        else:
            if (self.STOP_MODE_ACTIVATED or self.GO_MODE_ACTIVATED):

                if (self.STOP_MODE_ACTIVATED and (Traffic_State=="Go")): # 빨간불 -> 초록불로 바뀐 경우
                    self.STOP_MODE_ACTIVATED = False
                    self.GO_MODE_ACTIVATED = True

                elif(self.STOP_MODE_ACTIVATED): # 빨간불
                    action = "stop"
                elif(self.GO_MODE_ACTIVATED): # 초록불
                    ## 교차로를 건너기 위해 200 iteration 동안 강제 직진하게 하였음 ##
                    action = "go_straight"                 
                    if(self.TrafficLight_iterations==200):
                        self.GO_MODE_ACTIVATED = False
                        print("Interchange Crossed !!!")
                        self.TrafficLight_iterations = 0 #Reset

                    self.TrafficLight_iterations = self.TrafficLight_iterations + 1
            
        return action
    
    def process_signal(self,bounding_boxes):
        if bounding_boxes.detections == []: # 객체가 인식되지 않았을때는 초기화
            self.closest_signal = "Unknown"


        signal_min_distance = float("inf") # 초기 최솟값은 무한대로 설정
        

        for bbox in bounding_boxes.detections: 
            if bbox.results[0].id == "thirty" or bbox.results[0].id == "sixty" or bbox.results[0].id == "ninety" or bbox.results[0].id == "stop" or bbox.results[0].id == "left" or bbox.results[0].id == "right": # 객체 class가 표지판 class인 경우
                print(f'Signal Detected: {bbox.results[0].id}')
                signal_distance = self.calc_signal_dis(bbox) # 표지판까지 거리 계산
                if signal_distance < signal_min_distance:
                    signal_min_distance = signal_distance # 가장 가까운 표지판 거리 계산
                    if signal_min_distance <= 4: # 가장 가까운 표지판까지의 거리가 4미터 이내일 때 class 저장 및 tracking 모드 활성화
                        self.closest_signal = bbox.results[0].id
                        self.Mode = "Tracking"

        
        if self.closest_signal == "thirty":
            action = "30"
        elif self.closest_signal == "sixty":
            action = "60"
        elif self.closest_signal == "ninety":
            action = "90"
        elif self.closest_signal == "stop":
            action = "stop"
            print("Stop sign detected!!!")
        elif self.closest_signal == "left":
            self.direction = "left"

        elif self.closest_signal == "right":
            self.direction = "right"
        else:
            action = "none" # default 값은 none으로 설정
        if self.closest_signal == "left" or self.closest_signal == "right" or self.prev_Mode_Turn == "Tracking":
            action = self.process_Turn()
        print(f'Closest_Signal: {self.closest_signal}')
        self.Mode = "Detection"
        
        return action
    
    
    def process_Turn(self):
        action = "none"
        # 회전 수행
        
        if ( (self.prev_Mode_Turn =="Detection") and (self.Mode=="Tracking")): # 표지판이 처음 인식되었을 때
            self.prev_Mode_Turn = "Tracking"

        elif ( (self.prev_Mode_Turn =="Tracking") and (self.Mode=="Detection")): # 표지판이 인식되다가 인식되지 않을 때
            action = self.direction # 표지판이 인식되지 않는 시점에서부터 회전 action 작동
            self.prev_Mode_Turn = "Detection"
        
        return action


    def calc_signal_dis(self, bbox):
        # 표지판 까지의 거리를 계산하는 함수

        H_camera = 150  # 카메라의 실제 높이(cm)
        focal_length = 687 # 카메라의 초점 거리(픽셀)

        signboard_length = 1.10 # 표지판 헤드의 높이
        signcolumn_length = 1.35 # 기둥의 길이
        signboard_pixel = bbox.bbox.size_y # 표지판 헤드의 바운딩 박스 높이

        signal_height_in_pixels = (signboard_length + signcolumn_length) / signboard_length * signboard_pixel # 표지판 전체의 높이 픽셀

        distance = (H_camera * focal_length) / signal_height_in_pixels #표지판까지의 거리 계산
        distance_m = distance / 100  # cm를 m로 변환

        return distance_m



def main(args=None):
    rclpy.init(args=args)
    node = PlanningNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
