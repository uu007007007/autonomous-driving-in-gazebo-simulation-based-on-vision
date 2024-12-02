from ament_index_python.packages import get_package_share_directory 
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

  package_dir=get_package_share_directory('self_driving_car_pkg')

  return LaunchDescription([
        
        # control 노드
        Node(
          package='self_driving_car_pkg',
          executable='control_node',
          name='control_node',
          output='screen' # log 출력
        ),
    ])