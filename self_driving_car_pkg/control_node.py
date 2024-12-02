#!/usr/bin/env python3

import cv2
from geometry_msgs.msg import Twist
from rclpy.node import Node
from cv_bridge import CvBridge
from my_msgs.msg import LaneMsg
from std_msgs.msg import String
from sensor_msgs.msg import Image
import rclpy
from numpy import interp
from collections import deque

# from .Drive_Bot import Car, Debugging

import os
import sys

# print문 buffer 비활성화 -> print문 바로 출력
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

class Control(Node):
    def __init__(self):

        super().__init__('control_node')
        self.subscriber = self.create_subscription(LaneMsg,'/lane_state',self.lane_callback,10) # 차선 인식 결과
        self.subscriber = self.create_subscription(String,'/action',self.action_callback,10) # planning 결과
        self.image_sub = self.create_subscription(Image,'/camera/image_raw',self.image_callback,10) # 카메라 이미지
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 40) # 제어값 publish
        timer_period = 0.03;self.timer = self.create_timer(timer_period, self.send_cmd_vel)

        self.velocity = Twist()
        self.bridge   = CvBridge() # converting ros images to opencv data

        # 노드 변수 초기화
        self.distance = None
        self.Curvature = None        
        self.action = None
        self.img = None
        self.img_flag = None

        # 회전 수행 변수
        self.turn_iterations = 0
        self.Frozen_Angle = 0
        self.Activat_LTurn = False
        self.Activat_RTurn = False

        self.prev_Action = "none" # 이전 행동 저장

        # 제어 변수
        self.angle_queue = deque(maxlen=10)
        self.prev_speed = 0 # 이전 속도 저장
        self.speed = 30
        self.angle = 0
        
    ################### callback ###################

    def lane_callback(self, data):
        self.distance = data.distance
        self.Curvature = data.curvature
    
    def image_callback(self, frame):
        frame = self.bridge.imgmsg_to_cv2(frame,'bgr8')
        img = frame[0:640, 238:1042]
        self.img = cv2.resize(img, (320, 240))
        self.img_flag = True # 이미지가 들어왔는지 확인

    def action_callback(self, data):
        self.action = data.data

    #################################################


    def send_cmd_vel(self):
        Current_State = [self.distance, self.Curvature, self.img_flag, self.action]
        print(Current_State)
        if all(ele != None for ele in Current_State): # current_state 값이 모두 할당되어야 실행
            self.process_data(Current_State)
            print(f'action : {self.action}, speed : {self.speed}, angle : {self.angle}')
            self.publisher.publish(self.velocity)

    def process_data(self, Current_State):
        # self.Debug.setDebugParameters()
        [Distance, Curvature, img_flag, Action] = Current_State
        # Angle,Speed,img = self.Car.driveCar(Current_State)

        # stop에서 해제 되는경우 이전 속도 값 회복
        if self.prev_Action == "stop" and Action != "stop":
            self.speed = self.prev_speed

        # 차선 인식 주행
        if Action == "none":
            if((Distance != -1000) and (Curvature != -1000)):

                # [NEW]: Very Important: Minimum Sane Distance that a car can be from the perfect lane to follow is increased to half its fov.
                #                        This means sharp turns only in case where we are way of target XD
                self.angle_of_car = self.follow_Lane(int(self.img.shape[1]/2), Distance,Curvature)
            # Rolling average applied to get smoother steering angles for robot
            self.angle_queue.append(self.angle_of_car)
            self.angle_of_car = (sum(self.angle_queue)/len(self.angle_queue))
            self.angle = self.angle_of_car
        
        # 회전 표지판 인식
        elif Action == "left" or Action == "right":
            if Action == "left":
                self.Activat_LTurn = True
            else:
                self.Activat_RTurn = True
            print("Turn Activated!!!!!!!")

        elif Action == "go_straight":
            self.angle = 0

        elif Action == "stop":
            # stop이 처음 실시 되었을때 이전 속도 저장
            if self.prev_Action != "stop":
                self.prev_speed = self.speed
            self.speed = 0
        
        elif Action == "30":
            self.speed = 30

        elif Action == "60":
            self.speed = 60
        
        elif Action == "90":
            self.speed = 90

        


        # 회전 주행
        if self.Activat_LTurn or self.Activat_RTurn:
            if self.Activat_LTurn:
                print("Left Turn Processing.....")
            if self.Activat_RTurn:
                print("Right Turn Processing.....")
            print(f'iter: {self.turn_iterations}, angle: {self.Frozen_Angle}')
            self.speed = 50
            if ( ((self.turn_iterations % 20 ) ==0) and (self.turn_iterations>100) ):
                if self.Activat_LTurn:
                    self.Frozen_Angle = self.Frozen_Angle +7 # Move left by 7 degree
                elif self.Activat_RTurn:
                    self.Frozen_Angle = self.Frozen_Angle -7 # Move right by 7 degree
                
                
            if(self.turn_iterations==250):
                self.Activat_LTurn = False
                self.Activat_RTurn = False
                self.turn_iterations = 0
                self.Frozen_Angle = 0
                print("Turn finished!!!")
            self.turn_iterations = self.turn_iterations + 1

            
            self.angle = self.Frozen_Angle

        # 이전 action 저장
        self.prev_Action = Action

        
        
        # 속도와 각도를 시뮬레이션에서는 사용하는 값으로 맵핑
        Angle=interp(self.angle,[-60,60],[0.8,-0.8]) # 각도: -60~60 -> -0.8~0.8
        if (self.speed!=0):
            Speed=interp(self.speed,[30,90],[1,2]) # 속도: 30~90 -> 1~2
        elif (self.speed == 0):
            Speed = self.speed
            
        # 제어 변수 메세지 타입으로 정의
        self.velocity.angular.z = float(Angle)
        self.velocity.linear.x = float(Speed)


    def follow_Lane(self,Max_Sane_dist,distance,curvature):

        # [NEW]: Turning at normal speed is not much of a problem in simulation
        IncreaseTireSpeedInTurns = False
        
        Max_turn_angle_neg = -90
        Max_turn_angle = 90

        CarTurn_angle = 0

        if( (distance > Max_Sane_dist) or (distance < (-1 * Max_Sane_dist) ) ):
            # Max sane distance reached ---> Max penalize (Max turn Tires)
            if(distance > Max_Sane_dist):
                #Car offseted left --> Turn full wheels right
                CarTurn_angle = Max_turn_angle + curvature
            else:
                #Car Offseted right--> Turn full wheels left
                CarTurn_angle = Max_turn_angle_neg + curvature
        else:
            # Within allowed distance limits for car and lane
            # Interpolate distance to Angle Range
            Turn_angle_interpolated = interp(distance,[-Max_Sane_dist,Max_Sane_dist],[-90,90])
            #[NEW]: Modified to calculate carturn_angle based on following criteria
            #             65% turn suggested by distance to the lane center + 35 % how much the lane is turning
            CarTurn_angle = 1.2 * (0.65*Turn_angle_interpolated) + (0.35*curvature) # 0.35 => 0.5

        # Handle Max Limit [if (greater then either limits) --> set to max limit]
        if( (CarTurn_angle > Max_turn_angle) or (CarTurn_angle < (-1 *Max_turn_angle) ) ):
            if(CarTurn_angle > Max_turn_angle):
                CarTurn_angle = Max_turn_angle
            else:
                CarTurn_angle = -Max_turn_angle

        #angle = CarTurn_angle
        # [NEW]: Increase car turning capability by 30 % to accomodate sharper turns
        angle = interp(CarTurn_angle,[-90,90],[-60,60])


        
        return angle
    

def main(args=None):
  rclpy.init(args=args)
  node = Control()
  rclpy.spin(node)
  rclpy.shutdown()

if __name__ == '__main__':
	main()