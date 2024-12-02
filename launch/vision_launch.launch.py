import os
from ament_index_python.packages import get_package_share_directory 
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

  package_dir=get_package_share_directory('self_driving_car_pkg')

  # Yolo 모델 지정
  model = LaunchConfiguration("model")
  model_arg = DeclareLaunchArgument(
    "model",
    default_value=os.path.join(
        package_dir,
        "..","..","..","..","src",
        "self_driving_car_pkg",
        "best.pt"
    )
  )
  # 트래커 지정
  tracker = LaunchConfiguration("tracker")
  tracker_arg = DeclareLaunchArgument(
    "tracker",
    default_value="bytetrack.yaml"
  )
  # GPU/CPU 지정
  device = LaunchConfiguration("device")
  device_arg = DeclareLaunchArgument(
    "device",
    default_value="cuda:0"
  )
  # Yolo 활성화 지정
  enable = LaunchConfiguration("enable")
  enable_arg = DeclareLaunchArgument(
    "enable",
    default_value="true"
  )
  # threshold 지정
  threshold = LaunchConfiguration("threshold")
  threshold_arg = DeclareLaunchArgument(
    "threshold",
    default_value="0.7"
  )

  return LaunchDescription([
        model_arg,
        tracker_arg,
        device_arg,
        enable_arg,
        threshold_arg,
        
        # computer vision 노드
        Node(
          package='self_driving_car_pkg',
          executable='lane_detection_node',
          name='lane_detection_node',
          output='screen'
        ),
        # computer vision 노드
        Node(
          package='self_driving_car_pkg',
          executable='trafficlight_detection_node',
          name='trafficlight_detection_node',
          output='screen'
        ),
        # Yolo 노드
        Node(
          package='self_driving_car_pkg',
          executable='yolov8_node',
          name='yolov8_node',
          parameters=[{"model": model,
                       "tracker": tracker,
                       "device": device,
                       "enable": enable,
                       "threshold": threshold}]
        )
    ])