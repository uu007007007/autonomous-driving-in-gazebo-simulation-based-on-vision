from ament_index_python.packages import get_package_share_directory 
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

  package_dir=get_package_share_directory('self_driving_car_pkg')

  return LaunchDescription([
        
        # planning 노드
        Node(
          package='self_driving_car_pkg',
          executable='planning_node',
          name='planning_node',
          output='screen' # planning 노드 log만 출력
        ),
        # trigger 노드
        Node(
          package='self_driving_car_pkg',
          executable='trigger_node',
          name='trigger_node'
        ),
    ])