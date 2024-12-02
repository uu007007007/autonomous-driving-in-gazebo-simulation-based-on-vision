import cv2
import torch
import random
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from cv_bridge import CvBridge
import sys
print(sys.path)

from ultralytics import YOLO
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose, Detection2DArray
from std_srvs.srv import SetBool
import time  # Time module for FPS calculation


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class Yolov8Node(Node):

    def __init__(self) -> None:
        super().__init__("yolov8_node")

        # Parameters
        self.declare_parameter("model", "/home/uu007007007/project_ws/src/self_driving_car_pkg/gazebo_yolo.pt") # your trained model, default yolov8m.pt
        model = self.get_parameter("model").get_parameter_value().string_value

        self.declare_parameter("tracker", "bytetrack.yaml")
        tracker = self.get_parameter("tracker").get_parameter_value().string_value

        self.declare_parameter("device", "cuda:0")  # Set to GPU
        device = self.get_parameter("device").get_parameter_value().string_value

        if device == "cuda:0" and not torch.cuda.is_available():
            self.get_logger().warn("CUDA is not available. Switching to CPU.")
            device = "cpu"

        self.declare_parameter("threshold", 0.7)
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value

        self.declare_parameter("enable", True)
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value

        self._class_to_color = {}
        self.cv_bridge = CvBridge()
        self.tracker = self.create_tracker(tracker)
        self.yolo = YOLO(model)
        self.yolo.fuse()
        self.yolo.to(device)  # Move model to the device

        # Topics
        self._pub = self.create_publisher(Detection2DArray, "detections", 10)
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
        self._sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_cb, qos_profile_sensor_data
        )

        # Services
        self._srv = self.create_service(SetBool, "enable", self.enable_cb)

        # FPS and inference time tracking variables
        self.frame_count = 0
        self.start_time = time.time()

    def create_tracker(self, tracker_yaml):
        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap")  # for linear_assignment

        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))

        assert cfg.tracker_type in ["bytetrack", "botsort"], \
            f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=1)
        return tracker

    def enable_cb(self, req: SetBool.Request, res: SetBool.Response) -> SetBool.Response:
        self.enable = req.data
        res.success = True
        return res

    def image_cb(self, msg: Image) -> None:
        if self.enable:
            # Convert image + predict
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)

            # Measure inference time
            inference_start_time = time.time()

            results = self.yolo.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=0.7,
                mode="track"
            )

            # Track
            det = results[0].boxes.cpu().numpy()

            person_detected = False  # 사람 감지 여부 초기화
            
            if len(det) > 0:
                im0s = self.yolo.predictor.batch[2]
                im0s = im0s if isinstance(im0s, list) else [im0s]

                tracks = self.tracker.update(det, im0s[0])
                if len(tracks) > 0:
                    results[0].update(boxes=torch.as_tensor(tracks[:, :-1]))

            # Create detections message
            detections_msg = Detection2DArray()
            detections_msg.header = msg.header

            results = results[0].cpu()

            person_detected = False

            for b in results.boxes:
                label = self.yolo.names[int(b.cls)]
                score = float(b.conf)
                if score < self.threshold:
                    continue

                # # Object detect    
                # if label == "person":
                #     person_detected = True
                #     self.get_logger().info("Person detected!")
                #     self.car.stop()

            
                # else:
                #     self.car.driveCar(cv_image)

                detection = Detection2D()
                box = b.xywh[0]

                # Get box values
                detection.bbox.center.x = float(box[0])
                detection.bbox.center.y = float(box[1])
                detection.bbox.size_x = float(box[2])
                detection.bbox.size_y = float(box[3])

                # Get track ID
                track_id = -1
                if not b.id is None:
                    track_id = int(b.id)

                # Get hypothesis
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = label
                hypothesis.score = score
                detection.results.append(hypothesis)

                # Draw boxes for debug
                if label not in self._class_to_color:
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    self._class_to_color[label] = (r, g, b)
                color = self._class_to_color[label]

                min_pt = (
                    round(detection.bbox.center.x - detection.bbox.size_x / 2.0),
                    round(detection.bbox.center.y - detection.bbox.size_y / 2.0)
                )
                max_pt = (
                    round(detection.bbox.center.x + detection.bbox.size_x / 2.0),
                    round(detection.bbox.center.y + detection.bbox.size_y / 2.0)
                )

                cv2.rectangle(cv_image, min_pt, max_pt, color, 4)

                label_text = "{} ({}) ({:.3f})".format(label, str(track_id), score)
                pos = (min_pt[0] + 5, min_pt[1] + 25)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cv_image, label_text, pos, font, 1, color, 1, cv2.LINE_AA)

                # Append message
                detections_msg.detections.append(detection)

            # Publish detections and debug image
            self._pub.publish(detections_msg)
            self._dbg_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, encoding=msg.encoding))
            
            # Measure FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                fps = self.frame_count / elapsed_time
                #self.get_logger().info(f"FPS: {fps:.2f}, Inference Time: {time.time() - inference_start_time:.4f} seconds")

        # Display the result image
        cv_image = cv2.resize(cv_image, (640, 360))
        cv2.imshow('result', cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)  # Allow the window to update

def main():
    rclpy.init()
    node = Yolov8Node()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()