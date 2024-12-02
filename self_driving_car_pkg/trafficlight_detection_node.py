import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from my_msgs.msg import TrafficMsg
from cv_bridge import CvBridge
import cv2
from .Detection.TrafficLights.TrafficLights_Detection import detect_TrafficLights


class TrafficLightDetectionNode(Node):
    def __init__(self):
        super().__init__('trafficlight_detection_node')

        # Initialize subscriber for camera feed
        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Initialize publisher for traffic light state
        self.state_publisher = self.create_publisher(TrafficMsg, '/traffic_state', 10)

        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()

        # Debugging flag
        self.debugging = True

    def image_callback(self, msg):
        """Callback function for processing incoming camera images."""
        try:
            # Convert ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Detect traffic light state
            traffic_light_state, close_proximity = detect_TrafficLights(frame, frame.copy())

            # Publish the traffic light state
            state_msg = TrafficMsg()
            state_msg.state = traffic_light_state
            state_msg.proximity = close_proximity
            self.state_publisher.publish(state_msg)

            # Log the state for debugging
            self.get_logger().info(f"Traffic Light State: {traffic_light_state}, Close Proximity: {close_proximity}")

            # Debug visualization
            if self.debugging:
                self.debug_visualization(frame, traffic_light_state)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def debug_visualization(self, frame, traffic_light_state):
        """Overlay traffic light detection results on the frame and display."""
        debug_frame = frame.copy()

        # Display traffic light state on the frame
        cv2.putText(
            debug_frame,
            f"Traffic Light: {traffic_light_state}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if traffic_light_state == "Go" else (0, 0, 255),
            2
        )

        # Show debug frame
        debug_frame = cv2.resize(debug_frame, (640, 360))
        cv2.imshow('Traffic Light Detection', debug_frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
