from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz'
    )
    
    # Get package share directory
    pkg_share = FindPackageShare('fast_calib')
    
    # Parameters file
    params_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'qr_params.yaml'
    ])
    
    # RViz config file
    rviz_config = PathJoinSubstitution([
        pkg_share,
        'rviz_cfg',
        'fast_livo2.rviz'
    ])
    
    # Multi fast calib node
    multi_fast_calib_node = Node(
        package='fast_calib',
        executable='multi_fast_calib',
        name='multi_fast_calib',
        parameters=[params_file],
        output='screen'
    )
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )
    
    return LaunchDescription([
        rviz_arg,
        multi_fast_calib_node,
        rviz_node
    ])