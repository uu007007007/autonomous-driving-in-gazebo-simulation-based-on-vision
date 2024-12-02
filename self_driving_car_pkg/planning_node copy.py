import rclpy
from rclpy.node import Node
from my_msgs.msg import TrafficMsg
from sensor_msgs.msg import String
from cv_bridge import CvBridge
from collections import deque
from .Detection.Lanes.Lane_Detection import detect_Lane
from vision_msgs.msg import Detection2DArray
import cv2
from .Drive_Bot import Debugging
import time

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('planning_node')

        # ROS 퍼블리셔 초기화
        self.action_pub = self.create_publisher(String, '/action', 10)

        # ROS 이미지 구독자 초기화
        self.traffic_sub = self.create_subscription(TrafficMsg,'traffic_state',self.traffic_callback,10)
        self.yolo_sub = self.create_subscription(Detection2DArray,'detections',self.yolo_callback,10)
        timer_period = 0.5;self.timer = self.create_timer(timer_period, self.planning_pub)

        self.TrafficLight_iterations = 0
        self.GO_MODE_ACTIVATED = False
        self.STOP_MODE_ACTIVATED = False
        # self.SIG_STOP_ACTIVTED = False
        # self.Stop_signal_iterations = 0

        self.traffic_state = None
        self.close_proximity = None
        self.bounding_boxes = None

        self.current_state = [self.traffic_state, self.close_proximity, self.bounding_boxes]



    def traffic_callback(self, data):
        self.traffic_state = data.state
        self.close_proximity = data.proximity



    def yolo_callback(self, data):
        self.bounding_boxes = data


    def planning_pub(self):
        if all(ele != None for ele in self.current_state): # current_state 값이 모두 할당되어야 실행
            action = self.process(self.current_state)
            action_msg = String()
            action_msg.data = action
            self.action_pub.publish(action_msg)

    def process(self, current_state):
        bounding_boxes = current_state[2]
        traffic_elements = current_state[0:2]
        # human_action = self.process_human(bounding_boxes)
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
        action = "none"
        if((Traffic_State == "Stop") and CloseProximity):
            action = "stop"
            self.STOP_MODE_ACTIVATED = True
        else:
            if (self.STOP_MODE_ACTIVATED or self.GO_MODE_ACTIVATED):

                if (self.STOP_MODE_ACTIVATED and (Traffic_State=="Go")):
                    self.STOP_MODE_ACTIVATED = False
                    self.GO_MODE_ACTIVATED = True

                elif(self.STOP_MODE_ACTIVATED):
                    action = "stop"
                elif(self.GO_MODE_ACTIVATED):
                    # 교차로를 건너기 위해 200 iteration 동안 강제 직진하게 하였음
                    action = "go_straight"                 
                    if(self.TrafficLight_iterations==200):
                        self.GO_MODE_ACTIVATED = False
                        print("Interchange Crossed !!!")
                        self.TrafficLight_iterations = 0 #Reset

                    self.TrafficLight_iterations = self.TrafficLight_iterations + 1
            
        return action
    
    def process_signal(self,bounding_boxes):
        action = "none" # default 값은 none으로 설정
        signal_min_distance = float("inf") # 초기 최솟값은 무한대로 설정
        closest_signal = None
        for bbox in bounding_boxes.detections: 
            if bbox.results.id == "thirty" or bbox.results.id == "sixty" or bbox.results.id == "ninety" or bbox.results.id == "stop":
                signal_distance = self.calc_signal_dis(bbox) # 표지판까지 거리 계산
                if signal_distance < signal_min_distance:
                    signal_min_distance = signal_distance
                    closest_signal = bbox.results.id

        if signal_min_distance <= 4:
            if closest_signal == "thirty":
                action = "30"
            elif closest_signal == "sixty":
                action = "60"
            elif closest_signal == "ninety":
                action = "90"
            elif closest_signal == "stop":
                # action = self.stop_process(self.SIG_STOP_ACTIVTED)
                action = "stop"
        return action
        # if self.SIG_STOP_ACTIVTED:
        #     self.Stop_signal_iterations += 1
    # def stop_process(self):
    #     if self.SIG_STOP_ACTIVTED == False:
    #         self.SIG_STOP_ACTIVTED = True
    #         return "stop"
    #     else:
            
    #         if self.Stop_signal_iterations == 200:
    #             self.SIG_STOP_ACTIVTED = False
    #             return 

        

    # def process_human(self,bounding_boxes):
    #     human_min_distance = float("inf")
    #     for bbox in bounding_boxes.detections:
    #         if bbox.results.id == "human":
    #             human_distance = self.calc_human_dis(bbox)
    #             if human_distance < human_min_distance:
    #                 human_min_distance = human_distance

    #     if human_min_distance <= 4:
    #         return "stop"
    #     else:
    #         return "Go"


    # def calc_human_dis(self, bbox):
    #     H_camera = 150  # 카메라의 실제 높이(cm)
    #     focal_length = 687 # 카메라의 초점 거리(픽셀)

    #     # y축 거리 계산 (사람의 바운딩 박스 높이)
    #     person_height_in_pixels = bbox.bbox.size_y

    #     # 거리 계산
    #     distance = (H_camera * focal_length) / person_height_in_pixels
    #     distance_m = distance / 100  # cm를 m로 변환

    #     return distance_m
    
    
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






    # def image_callback(self, msg):
    #     """
    #     ROS 이미지 콜백: 이미지를 처리하고 차선 상태를 계산 후 발행.

    #     Args:
    #         msg (sensor_msgs.msg.Image): ROS 이미지 메시지.
    #     """
    #     try:
    #         # ROS 이미지를 OpenCV 이미지로 변환
    #         frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    #         # 차선 감지 처리
    #         distance, curvature = self.process_frame(frame)

    #         if distance != -1000 and curvature != -1000:
    #             self.publish_lane_state(distance, curvature)
            
    #         cv2.waitKey(1)

    #     except Exception as e:
    #         self.get_logger().error(f"Failed to process image: {e}")

    # def process_frame(self, frame):
    #     """
    #     단일 프레임을 처리하여 차선을 감지하고 각도를 계산합니다.

    #     Args:
    #         frame (numpy.ndarray): 입력 이미지 프레임.

    #     Returns:
    #         tuple: 감지된 차선의 거리, 곡률.
    #     """
    #     img = frame[0:640, 238:1042]
    #     img = cv2.resize(img, (320, 240))

    #     distance, curvature = detect_Lane(img)

    #     return distance, curvature

    def publish_lane_state(self, distance, curvature):
        """
        차선 상태(거리 및 곡률)를 ROS 토픽으로 발행합니다.

        Args:
            distance (float): 차선 중심으로부터의 거리.
            curvature (float): 차선 곡률.
        """
        msg = LaneMsg()
        msg.distance = float(distance)
        msg.curvature = float(curvature)
        self.lane_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
