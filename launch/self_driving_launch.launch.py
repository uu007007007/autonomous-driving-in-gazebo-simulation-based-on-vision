from ament_index_python.packages import get_package_share_directory 
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

  package_dir=get_package_share_directory('self_driving_car_pkg')

  return LaunchDescription([
        
        # yolo 노드
        Node(
          package='self_driving_car_pkg',
          executable='yolov8_node',
          name='yolov8_node'
        ),

        # 정지선 노드
        Node(
          package='self_driving_car_pkg',
          executable='stopline_node',
          name='stopline_node'
        ),

        # lidar tracking 노드
        Node(
          package='self_driving_car_pkg',
          executable='lidar_node',
          name='lidar_node',
        ),

        # 장애물 회피 노드
        Node(
          package='self_driving_car_pkg',
          executable='object_avoid',
          name='object_avoid'
        ),

        # 제어 노드
        Node(
          package='self_driving_car_pkg',
          executable='computer_vision_node',
          name='computer_vision_node',
          output='screen'
        ),
    ])