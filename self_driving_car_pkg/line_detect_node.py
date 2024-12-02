import rclpy
from rclpy.node import Node
from my_msgs.msg import LaneMsg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque
from .Detection.Lanes.Lane_Detection import detect_Lane
import cv2

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # 조향 각도를 부드럽게 하기 위한 매개변수 설정
        self.angle_queue = deque(maxlen=10)

        # ROS 퍼블리셔 초기화
        self.lane_state_pub = self.create_publisher(LaneMsg, '/lane_state', 10)

        # ROS 이미지 구독자 초기화
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        # OpenCV-ROS 간 데이터 변환을 위한 브릿지
        self.bridge = CvBridge()

    def image_callback(self, msg):
        """
        ROS 이미지 콜백: 이미지를 처리하고 차선 상태를 계산 후 발행.

        Args:
            msg (sensor_msgs.msg.Image): ROS 이미지 메시지.
        """
        try:
            # ROS 이미지를 OpenCV 이미지로 변환
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 차선 감지 처리
            distance, curvature = self.process_frame(frame)

            if distance != -1000 and curvature != -1000:
                self.publish_lane_state(distance, curvature)
            
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def process_frame(self, frame):
        """
        단일 프레임을 처리하여 차선을 감지하고 각도를 계산합니다.

        Args:
            frame (numpy.ndarray): 입력 이미지 프레임.

        Returns:
            tuple: 감지된 차선의 거리, 곡률.
        """
        img = frame[0:640, 238:1042]
        img = cv2.resize(img, (320, 240))

        distance, curvature = detect_Lane(img)

        return distance, curvature

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
