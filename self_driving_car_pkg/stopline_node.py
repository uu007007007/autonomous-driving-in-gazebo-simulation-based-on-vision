import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import cv2
import numpy as np
from cv_bridge import CvBridge

class StopLineDetectionNode(Node):
    def __init__(self):
        super().__init__('stopline_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.publisher_ = self.create_publisher(Bool, '/stop_line_detected', 10)
        self.bridge = CvBridge()
        self.previous_detection = None
        self.get_logger().info('Stop Line Detection Node has been started.')

    def image_callback(self, msg):
        # ROS Image 메시지를 OpenCV 형식으로 변환
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 정지선 감지
        stop_line_detected = self.detect_StopLine()
        
        # 감지 결과 퍼블리시
        detection_msg = Bool()
        detection_msg.data = stop_line_detected
        self.publisher_.publish(detection_msg)
        
        # 이전 상태와 현재 상태가 다를 때만 로그 출력
        if self.previous_detection is None or stop_line_detected != self.previous_detection:
            if stop_line_detected:
                self.get_logger().info('정지선이 감지되었습니다.')
            else:
                self.get_logger().info('정지선이 감지되지 않았습니다.')
        
        # 현재 상태를 이전 상태로 업데이트
        self.previous_detection = stop_line_detected
        cv2.waitKey(1)

        

    def detect_StopLine(self, roi_top_left=(100, 200), roi_bottom_right=(400, 400)):
        stop_detected = False
        # 관심 영역 설정
        x1, y1 = roi_top_left
        x2, y2 = roi_bottom_right
        cv2.imshow('1', self.frame)
        height, width, _ = self.frame.shape
        # frame = frame[y1:y2, x1:x2]
        # print(height, width)
        top_left = (490, 400)
        bottom_left = (360,525)
        top_right = (width - top_left[0], top_left[1])
        bottom_right = (width - bottom_left[0], bottom_left[1])
        # print([top_left, top_right, bottom_right, bottom_left])
        src = np.float32([top_left, top_right, bottom_right, bottom_left])
        dst = np.float32([(0, 0), (width, 0), (width, height), (0,height)])

        mat = cv2.getPerspectiveTransform(src, dst)
        trans_img = cv2.warpPerspective(self.frame, mat, (width, height))
        trans_img = cv2.resize(trans_img, (640, 360))
        roi_img = trans_img[trans_img.shape[0] // 2:, :]

        # HSV로 색 추출
        hsvLower = np.array([0, 0, 200])    # 추출할 색의 하한(HSV) (색상, 채도, 명도)
        hsvUpper = np.array([0, 5, 255])    # 추출할 색의 상한(HSV)
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV) # 이미지를 HSV으로 변환
        hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSV에서 마스크를 작성
        roi_img = cv2.bitwise_and(roi_img, roi_img, mask=hsv_mask) # 원래 이미지와 마스크를 합성

        # cv2.imshow('trans_img', trans_img)
        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        # 블러 적용하여 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 엣지 검출
        edges = cv2.Canny(blurred, 50, 150)
        # 이미지의 하단 절반만 관심 영역으로 설정
        # roi = edges[frame.shape[0] // 2:, :]

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, None, 400, 100)
        
        if lines is not None:
            # print("검출된 직선 개수 : ", len(lines))
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(roi_img, (x1,y1), (x2, y2), (0,0,255), 2)
                # 분수로 기울기를 구하면 기울기가 0인경우 오류가 발생할 수 있음
                # 따라서 벡터를 이용해 x축 단위 벡터와의 사이각도를 구함
                vector = np.array([x2-x1, y2-y1])
                norm = np.linalg.norm(vector)
                v_unit = vector/norm
                # print("vector",vector)
                # print("norm", norm)
                # print("v_unit",v_unit)
                x_unit = (1, 0)
                theta = np.degrees(np.arccos(v_unit @ x_unit))
                # print(v_unit @ x_unit)
                # print(theta)
                if theta < 2:
                    stop_detected = True

                

        # # 관심 영역에서 컨투어 찾기
        # contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # for contour in contours:
        #     # 컨투어를 다각형으로 근사화
        #     approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        #     cv2.polylines(edges, [approx], True, (0, 255, 0), 2)
        #     # 다각형이 4개의 변을 가지면 정지선일 가능성 있음
        #     if len(approx) == 4:
        #         # 컨투어의 경계 상자 계산
        #         x, y, w, h = cv2.boundingRect(approx)
        #         aspect_ratio = w / float(h)
        #         # 가로 세로 비율이 정지선의 합리적인 범위 내에 있는지 확인
        #         if 6 < aspect_ratio < 10:
        #             return True
                

        
        cv2.imshow("polygon", roi_img)
        cv2.imshow("trans", trans_img)
        

        return stop_detected # True or False return

def main(args=None):
    rclpy.init(args=args)
    node = StopLineDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
