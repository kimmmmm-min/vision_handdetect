import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    realsense_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch', 'rs_launch.py')

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={
            'align_depth.enable': 'true',
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
        }.items(),
    )

    hand_detector_node = Node(
        package='hand_detector',
        executable='hand_detector_node',
        name='hand_detector_node',
        output='screen',
        # onnxruntime's global Environment::Initialize() does contrib-op
        # schema registration under pthread_once with a thread-safety bug
        # (heap corruption -> SIGABRT) that fires under CPU contention.
        # Starting this at the same instant as the RealSense driver's own
        # startup burst (device probe, sensor negotiation) makes it fire
        # reliably, so give the camera a few seconds' head start. respawn
        # is a safety net in case the race still gets hit anyway - once a
        # given process instance survives its first couple of seconds, it
        # runs indefinitely without issue.
        respawn=True,
        respawn_delay=1.0,
    )

    return LaunchDescription([
        realsense_launch,
        TimerAction(period=5.0, actions=[hand_detector_node]),
    ])
