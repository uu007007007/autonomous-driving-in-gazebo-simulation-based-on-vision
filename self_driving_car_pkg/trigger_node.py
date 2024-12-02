import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Bool

import os
import sys

# 비활성화
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

class TriggerNode(Node):
    def __init__(self):
        super().__init__('trigger_node')

        # 구독자: YOLO로부터 감지된 바운딩 박스 정보
        self.yolo_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.yolo_callback,
            10
        )

        # 퍼블리셔: 차량 행동 상태
        self.action_pub = self.create_publisher(Bool, '/trigger', 10)
        timer_period = 0.03;self.timer = self.create_timer(timer_period, self.trigger_pub)
        self.trigger = False
        # 파라미터
        self.stop_distance = 4.0  # 차량과 사람 사이의 안전 거리(m)

    def yolo_callback(self, msg):
        """
        YOLO에서 전달받은 감지 정보를 처리하여 사람과의 거리를 계산하고 멈춤 여부를 결정합니다.
        """
        human_min_distance = float("inf")

        # 감지된 객체 중 사람이 있는지 확인
        for detection in msg.detections:
            # 객체 클래스 ID가 'human'일 경우
            if detection.results[0].id == "human":
                # 사람과의 거리 계산
                human_distance = self.calc_human_distance(detection)
                self.get_logger().info(f"Human detected at {human_distance:.2f} meters.")
                if human_distance < human_min_distance:
                    human_min_distance = human_distance

        # 멈춤 또는 이동 행동 결정
        if human_min_distance <= self.stop_distance:
            self.trigger = True
        else:
            self.trigger = False

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

    def trigger_pub(self):
        """
        트리거 상태를 퍼블리시합니다.

        Args:
            trigger (bool): 트리거 상태 ("true" 또는 "false").
        """
        action_msg = Bool()
        action_msg.data = self.trigger
        self.action_pub.publish(action_msg)
        print(f"Trigger published: {self.trigger}")


def main(args=None):
    rclpy.init(args=args)
    node = TriggerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
