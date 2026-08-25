from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    hand_motion_overlay_node = Node(
        package='hand_detector',
        executable='hand_motion_overlay_node',
        name='hand_motion_overlay_node',
        output='screen',
    )

    return LaunchDescription([
        hand_motion_overlay_node,
    ])
