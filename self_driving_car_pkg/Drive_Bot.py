from .Detection.Lanes.Lane_Detection import detect_Lane
from .Detection.Signs.SignDetectionApi import detect_Signs
from .Detection.TrafficLights.TrafficLights_Detection import detect_TrafficLights
import cv2
from numpy import interp
from .config import config
from collections import deque

class Debugging:

    def __init__(self):
        self.TL_Created = False
        self.Lan_Created = False

    def nothing(self,x):
        pass

    cv2.namedWindow('CONFIG')

    enable_SatNav = 'Sat-Nav'
    cv2.createTrackbar(enable_SatNav, 'CONFIG',False,True,nothing)

    # creating (Engine) on/off trackbar 
    Motors = 'Engine'
    cv2.createTrackbar(Motors, 'CONFIG',False,True,nothing)

    # create switch for ON/OFF functionality
    debugging_SW = 'Debug'
    cv2.createTrackbar(debugging_SW, 'CONFIG',False,True,nothing)
    # create switch for ON/OFF functionality
    debuggingLane_SW = 'Debug Lane'
    cv2.createTrackbar(debuggingLane_SW, 'CONFIG',False,True,nothing)
    # create switch for ON/OFF functionality
    debuggingSigns_SW = 'Debug Sign'
    cv2.createTrackbar(debuggingSigns_SW, 'CONFIG',False,True,nothing)
    # create switch for ON/OFF functionality
    debuggingTL_SW = 'Debug TL'
    cv2.createTrackbar(debuggingTL_SW, 'CONFIG',False,True,nothing)


    def setDebugParameters(self):
        # get current positions of four trackbars
        # get current positions of trackbar
        # get current positions of four trackbars
        enable_SatNav = cv2.getTrackbarPos(self.enable_SatNav,'CONFIG')
        Motors = cv2.getTrackbarPos(self.Motors,'CONFIG')

        debug = cv2.getTrackbarPos(self.debugging_SW,'CONFIG')
        debugLane = cv2.getTrackbarPos(self.debuggingLane_SW,'CONFIG')
        debugSign = cv2.getTrackbarPos(self.debuggingSigns_SW,'CONFIG')
        debugTrafficLights = cv2.getTrackbarPos(self.debuggingTL_SW,'CONFIG')


        if enable_SatNav:
            config.enable_SatNav = True
        else:
            config.enable_SatNav = False

        # If trackbar changed modify engines_on config parameter
        if Motors:
            config.engines_on = True
        else:
            config.engines_on = False
            
        if debug:
            config.debugging = True
        else:
            config.debugging = False            
        if debugLane:
            config.debugging_Lane = True
        else:
            config.debugging_Lane = False    
        if debugSign:
            config.debugging_Signs = True
        else:
            config.debugging_Signs = False           
        if debugTrafficLights:
            config.debugging_TrafficLights = True
        else:
            config.debugging_TrafficLights = False

        if config.debugging_TrafficLights:
            
            debuggingTLConfig_SW = 'Debug Config'
            if not self.TL_Created:
                self.TL_Created = True
                cv2.namedWindow('CONFIG_TL')
                cv2.createTrackbar(debuggingTLConfig_SW, 'CONFIG_TL',False,True,self.nothing)

            debugTL_Config = cv2.getTrackbarPos(debuggingTLConfig_SW,'CONFIG_TL')

            if debugTL_Config:
                config.debugging_TL_Config = True
            else:
                config.debugging_TL_Config = False

        else:
            self.TL_Created = False
            cv2.destroyWindow('CONFIG_TL')

        
        if config.debugging_Lane:
            
            debuggingLANEConfig_SW = 'Debug (Stage)'
            if not self.Lan_Created:
                self.Lan_Created = True
                cv2.namedWindow('CONFIG_LANE')
                cv2.createTrackbar(debuggingLANEConfig_SW, 'CONFIG_LANE',0,3,self.nothing)

            debugLane_Config = cv2.getTrackbarPos(debuggingLANEConfig_SW,'CONFIG_LANE')

            if debugLane_Config == 0:
                config.debugging_L_ColorSeg = True
                config.debugging_L_Est = config.debugging_L_Cleaning = config.debugging_L_LaneInfoExtraction = False                    
            elif debugLane_Config == 1:
                config.debugging_L_Est = True
                config.debugging_L_ColorSeg = config.debugging_L_Cleaning = config.debugging_L_LaneInfoExtraction = False   
            elif debugLane_Config == 2:
                config.debugging_L_Cleaning = True
                config.debugging_L_ColorSeg = config.debugging_L_Est = config.debugging_L_LaneInfoExtraction = False   
            elif debugLane_Config == 3:
                config.debugging_L_LaneInfoExtraction = True
                config.debugging_L_ColorSeg = config.debugging_L_Est = config.debugging_L_Cleaning = False

        else:
            self.Lan_Created = False
            cv2.destroyWindow('CONFIG_LANE')        

class Control:

    def __init__(self):
        self.prev_Mode = "Detection"
        self.prev_Mode_LT = "Detection"
        self.prev_Mode_RT = "Detection"
        self.car_speed = 80
        self.angle_of_car = 0

        self.Left_turn_iterations = 0
        self.Frozen_Angle = 0
        self.Detected_LeftTurn = False
        self.Activat_LeftTurn = False

        self.Right_turn_iterations = 0
        self.Frozen_Angle_RT = 0
        self.Detected_RightTurn = False
        self.Activat_RightTurn = False

        self.TrafficLight_iterations = 0
        self.GO_MODE_ACTIVATED = False
        self.STOP_MODE_ACTIVATED = False

        # [NEW]: Lane의 부드러운 조향을 위해 Rolling Average 필터를 사용하기 위한 deque 멤버 변수 생성
        self.angle_queue = deque(maxlen=10)

    def follow_Lane(self, Max_Sane_dist, distance, curvature, Mode, Tracked_class):

        # [NEW]: 시뮬레이션에서는 정상 속도로 회전해도 큰 문제가 없다
        IncreaseTireSpeedInTurns = False

        if ((Tracked_class != 0) and (self.prev_Mode == "Tracking") and (Mode == "Detection")):
            if (Tracked_class == "speed_sign_30"):
                self.car_speed = 30
            elif (Tracked_class == "speed_sign_60"):
                self.car_speed = 60
            elif (Tracked_class == "speed_sign_90"):
                self.car_speed = 90
            elif (Tracked_class == "stop"):
                self.car_speed = 0

        self.prev_Mode = Mode  # 현재 모드를 이전 모드로 설정

        Max_turn_angle_neg = -90
        Max_turn_angle = 90

        CarTurn_angle = 0

        if ((distance > Max_Sane_dist) or (distance < (-1 * Max_Sane_dist))):
            # 최대 허용 거리를 초과한 경우 --> 최대 회전 (타이어를 최대한 돌림)
            if (distance > Max_Sane_dist):
                # 차가 왼쪽으로 치우침 --> 타이어를 최대한 오른쪽으로 회전
                CarTurn_angle = Max_turn_angle + curvature
            else:
                # 차가 오른쪽으로 치우침 --> 타이어를 최대한 왼쪽으로 회전
                CarTurn_angle = Max_turn_angle_neg + curvature
        else:
            # 차와 차선이 허용된 거리 내에 있는 경우
            # 거리를 각도 범위로 보간
            Turn_angle_interpolated = interp(distance, [-Max_Sane_dist, Max_Sane_dist], [-90, 90])
            # [NEW]: 거리 중심으로부터 65% + 차선 곡률을 기준으로 35%의 각도로 조향각을 계산
            CarTurn_angle = (0.65 * Turn_angle_interpolated) + (0.35 * curvature)

        # 최대 한계 처리 [각도가 한계를 초과한 경우, 최대 한계로 설정]
        if ((CarTurn_angle > Max_turn_angle) or (CarTurn_angle < (-1 * Max_turn_angle))):
            if (CarTurn_angle > Max_turn_angle):
                CarTurn_angle = Max_turn_angle
            else:
                CarTurn_angle = -Max_turn_angle

        # [NEW]: 더 날카로운 회전을 수용하기 위해 자동차 회전 능력을 30% 증가시킴
        angle = interp(CarTurn_angle, [-90, 90], [-60, 60])

        curr_speed = self.car_speed

        if (IncreaseTireSpeedInTurns and (Tracked_class != "left_turn") and (Tracked_class != "right_turn")):
            if (angle > 30):
                car_speed_turn = interp(angle, [30, 45], [80, 100])
                curr_speed = car_speed_turn
            elif (angle < -30):
                car_speed_turn = interp(angle, [-45, -30], [100, 80])
                curr_speed = car_speed_turn

        return angle, curr_speed

    def Obey_LeftTurn(self, Angle, Speed, Mode, Tracked_class):

        if (Tracked_class == "left_turn"):

            Speed = 50

            if ((self.prev_Mode_LT == "Detection") and (Mode == "Tracking")):
                self.prev_Mode_LT = "Tracking"
                self.Detected_LeftTurn = True

            elif ((self.prev_Mode_LT == "Tracking") and (Mode == "Detection")):
                self.Detected_LeftTurn = False
                self.Activat_LeftTurn = True

                if (((self.Left_turn_iterations % 20) == 0) and (self.Left_turn_iterations > 100)):
                    self.Frozen_Angle = self.Frozen_Angle - 7  # 왼쪽으로 1도씩 이동
                if (self.Left_turn_iterations == 250):
                    self.prev_Mode_LT = "Detection"
                    self.Activat_LeftTurn = False
                    self.Left_turn_iterations = 0
                    self.Frozen_Angle = 0
                self.Left_turn_iterations = self.Left_turn_iterations + 1

                if (self.Activat_LeftTurn or self.Detected_LeftTurn):
                    # 이전에 저장된 경로를 따름
                    Angle = self.Frozen_Angle

        return Angle, Speed, self.Detected_LeftTurn, self.Activat_LeftTurn

    def Obey_RightTurn(self, Angle, Speed, Mode, Tracked_class):

        if (Tracked_class == "right_turn"):

            Speed = 50

            if ((self.prev_Mode_RT == "Detection") and (Mode == "Tracking")):
                self.prev_Mode_RT = "Tracking"
                self.Detected_RightTurn = True

            elif ((self.prev_Mode_RT == "Tracking") and (Mode == "Detection")):
                self.Detected_RightTurn = False
                self.Activat_RightTurn = True

                if (((self.Right_turn_iterations % 20) == 0) and (self.Right_turn_iterations > 60)):
                    self.Frozen_Angle_RT = self.Frozen_Angle_RT + 12  # 오른쪽으로 1도씩 이동
                if (self.Right_turn_iterations == 280):
                    self.prev_Mode_RT = "Detection"
                    self.Activat_RightTurn = False
                    self.Right_turn_iterations = 0
                    self.Frozen_Angle_RT = 0
                self.Right_turn_iterations = self.Right_turn_iterations + 1

                if (self.Activat_RightTurn or self.Detected_RightTurn):
                    # 이전에 저장된 경로를 따름
                    Angle = self.Frozen_Angle_RT

        return Angle, Speed, self.Detected_RightTurn, self.Activat_RightTurn

    def OBEY_TrafficLights(self, a, b, Traffic_State, CloseProximity):

        if ((Traffic_State == "Stop") and CloseProximity):
            b = 0  # 정지 상태
            self.STOP_MODE_ACTIVATED = True
        else:
            if (self.STOP_MODE_ACTIVATED or self.GO_MODE_ACTIVATED):

                if (self.STOP_MODE_ACTIVATED and (Traffic_State == "Go")):
                    self.STOP_MODE_ACTIVATED = False
                    self.GO_MODE_ACTIVATED = True

                elif (self.STOP_MODE_ACTIVATED):
                    b = 0

                elif (self.GO_MODE_ACTIVATED):
                    a = 0.0
                    if (self.TrafficLight_iterations == 350):
                        self.GO_MODE_ACTIVATED = False
                        print("교차로를 통과했습니다!!!")
                        self.TrafficLight_iterations = 0  # 초기화

                    self.TrafficLight_iterations = self.TrafficLight_iterations + 1
        return a, b

    def drive_car(self, Current_State, Inc_TL, Inc_LT, Inc_RT):

        [Distance, Curvature, frame_disp, Mode, Tracked_class, Traffic_State, CloseProximity] = Current_State

        current_speed = 0

        if ((Distance != -1000) and (Curvature != -1000)):

            # [NEW]: 자동차가 완벽한 차선을 따를 수 있는 최소 허용 거리가 시야각의 절반으로 증가됨
            self.angle_of_car, current_speed = self.follow_Lane(int(frame_disp.shape[1] / 2), Distance, Curvature, Mode, Tracked_class)
        # [NEW]: 원래의 조향각과 Rolling Average를 통한 부드러운 조향각을 추적
        config.angle_orig = self.angle_of_car
        # Rolling Average를 적용하여 부드러운 조향각 계산
        self.angle_queue.append(self.angle_of_car)
        self.angle_of_car = (sum(self.angle_queue) / len(self.angle_queue))
        config.angle = self.angle_of_car
        if Inc_LT:
            self.angle_of_car, current_speed, Detected_LeftTurn, Activat_LeftTurn = self.Obey_LeftTurn(self.angle_of_car, current_speed, Mode, Tracked_class)
        else:
            Detected_LeftTurn = False
            Activat_LeftTurn = False

        if Inc_RT:
            self.angle_of_car, current_speed, Detected_RightTurn, Activat_RightTurn = self.Obey_RightTurn(self.angle_of_car, current_speed, Mode, Tracked_class)
        else:
            Detected_RightTurn = False
            Activat_RightTurn = False

        if Inc_TL:
            self.angle_of_car, current_speed = self.OBEY_TrafficLights(self.angle_of_car, current_speed, Traffic_State, CloseProximity)

        return self.angle_of_car, current_speed, Detected_LeftTurn, Activat_LeftTurn, Detected_RightTurn, Activat_RightTurn

class Car:
    def __init__(self, Inc_TL=True, Inc_LT=True, Inc_RT=True):

        self.Control_ = Control()
        self.Inc_TL = Inc_TL
        self.Inc_LT = Inc_LT
        self.Inc_RT = Inc_RT
        # [NEW]: 표지판 및 신호등 감지의 현재 상태를 추적하기 위한 컨테이너
        self.Tracked_class = "Unknown"
        self.Traffic_State = "Unknown"

    def display_state(self, frame_disp, angle_of_car, current_speed, Tracked_class, Traffic_State, Detected_LeftTurn, Activat_LeftTurn, Detected_RightTurn, Activat_RightTurn):

        ###################################################  제어 상태 디스플레이 ####################################

        if (angle_of_car < -10):
            direction_string = "[ Left ]"
            color_direction = (120, 0, 255)
        elif (angle_of_car > 10):
            direction_string = "[ Right ]"
            color_direction = (120, 0, 255)
        else:
            direction_string = "[ Straight ]"
            color_direction = (0, 255, 0)

        if (current_speed > 0):
            direction_string = "Moving --> " + direction_string
        else:
            color_direction = (0, 0, 255)

        cv2.putText(frame_disp, str(direction_string), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, color_direction, 1)

        # angle_speed_str = "[ Angle ,Speed ] = [ " + str(int(angle_of_car)) + "deg ," + str(int(current_speed)) + "mph ]"
        # cv2.putText(frame_disp, str(angle_speed_str), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1)

        cv2.putText(frame_disp, "Traffic Light State = [ " + Traffic_State + " ] ", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.35, 255)

        if (Tracked_class == "left_turn" or Tracked_class == "right_turn"):
            font_Scale = 0.32
            if (Detected_LeftTurn or Detected_RightTurn):
                Tracked_class = Tracked_class + " : Detected { True } "
            else:
                Tracked_class = Tracked_class + " : Activated { " + str(Activat_LeftTurn or Activat_RightTurn) + " } "
        else:
            font_Scale = 0.37
        cv2.putText(frame_disp, "Sign Detected ==> " + str(Tracked_class), (20, 60), cv2.FONT_HERSHEY_COMPLEX, font_Scale, (0, 255, 255), 1)

    def driveCar(self, frame):

        img = frame[0:640, 238:1042]
        img = cv2.resize(img, (320, 240))

        img_orig = img.copy()

        distance, Curvature = detect_Lane(img)

        if self.Inc_TL:
            Traffic_State, CloseProximity = detect_TrafficLights(img_orig.copy(), img)
        else:
            Traffic_State = "Unknown"
            CloseProximity = False

        Mode, Tracked_class = detect_Signs(img_orig, img)

        Current_State = [distance, Curvature, img, Mode, Tracked_class, Traffic_State, CloseProximity]

        Angle, Speed, Detected_LeftTurn, Activat_LeftTurn, Detected_RightTurn, Activat_RightTurn = self.Control_.drive_car(Current_State, self.Inc_TL, self.Inc_LT, self.Inc_RT)

        # [NEW]: 현재 상태 변수 업데이트
        self.Tracked_class = Tracked_class
        self.Traffic_State = Traffic_State

        self.display_state(img, Angle, Speed, Tracked_class, Traffic_State, Detected_LeftTurn, Activat_LeftTurn, Detected_RightTurn, Activat_RightTurn)

        # [NEW]: 증가된 자동차 조향 범위를 증가된 모터 회전 각도로 보간
        # [현실 세계 각도 및 속도 ===>> ROS 자동차 제어 범위로 변환]
        Angle = interp(Angle, [-60, 60], [0.8, -0.8])
        if (Speed != 0):
            Speed = interp(Speed, [30, 90], [1, 2])

        Speed = float(Speed)

        return Angle, Speed, img
