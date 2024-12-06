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
        self.image_pub = self.create_publisher(Image, "stopline_image", 10)
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

        

    def detect_StopLine(self, top_left = (490, 400), bottom_left = (360,525)):
        stop_detected = False
        height, width, _ = self.frame.shape
        
        top_right = (width - top_left[0], top_left[1])
        bottom_right = (width - bottom_left[0], bottom_left[1])
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

        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        # 블러 적용하여 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 엣지 검출
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, None, 400, 100)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(roi_img, (x1,y1), (x2, y2), (0,0,255), 2)
                # 분수로 기울기를 구하면 기울기가 0인경우 오류가 발생할 수 있음
                # 따라서 벡터를 이용해 x축 단위 벡터와의 사이각도를 구함
                vector = np.array([x2-x1, y2-y1])
                norm = np.linalg.norm(vector)
                v_unit = vector/norm
                x_unit = (1, 0)
                theta = np.degrees(np.arccos(v_unit @ x_unit))
                if theta < 2:
                    stop_detected = True

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(roi_img, encoding="bgr8")) # 정지선 이미지 pub

        return stop_detected # True or False return

def main(args=None):
    rclpy.init(args=args)
    node = StopLineDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
