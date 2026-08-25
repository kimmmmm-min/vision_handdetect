import array
import math
import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import message_filters
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo

from hand_detector_msgs.msg import HandLandmarks, HandLandmarksArray
from hand_detector.onnx_hand import PalmDetection, HandLandmark
from hand_detector.onnx_hand.utils import rotate_and_crop_rectangle

NUM_LANDMARKS = 21
# Standard MediaPipe hand-landmark skeleton (wrist=0, thumb=1-4,
# index=5-8, middle=9-12, ring=13-16, pinky=17-20).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

DEFAULT_TRT_CACHE_DIR = os.path.join(
    os.path.expanduser('~'), '.cache', 'hand_detector_trt')


def _handedness_from_3d(landmarks):
    """Determine left/right from real 3D landmark geometry rather than 2D
    image appearance.

    Uses the scalar triple product of (wrist->index_mcp) x (wrist->pinky_mcp)
    . (wrist->thumb_cmc). For any *rigid rotation* of a hand (e.g. turning
    it to show the back instead of the palm) this sign is invariant - only
    an actual mirror image (a genuinely different, opposite hand) flips it.
    That makes it robust to the palm/back-of-hand viewing angle in a way a
    2D-appearance classifier structurally cannot be.

    Returns HandLandmarks.LEFT/RIGHT, or None if the required points don't
    have valid (non-NaN) depth this frame.
    """
    wrist, thumb_cmc, index_mcp, pinky_mcp = (
        landmarks[0], landmarks[1], landmarks[5], landmarks[17])
    for p in (wrist, thumb_cmc, index_mcp, pinky_mcp):
        if math.isnan(p.x) or math.isnan(p.y) or math.isnan(p.z):
            return None

    w = np.array([wrist.x, wrist.y, wrist.z])
    t = np.array([thumb_cmc.x, thumb_cmc.y, thumb_cmc.z]) - w
    i = np.array([index_mcp.x, index_mcp.y, index_mcp.z]) - w
    p = np.array([pinky_mcp.x, pinky_mcp.y, pinky_mcp.z]) - w
    triple = float(np.dot(np.cross(i, p), t))
    # Sign convention calibrated against real hardware (the armchair-
    # derived assumption was backwards - real-hand test showed both
    # hands swapped, a pure global flip, which is expected: getting any
    # one of the anatomical sign assumptions backwards flips the overall
    # sign uniformly for every orientation, so this flip doesn't undermine
    # the front/back rotation-invariance itself, only the label mapping).
    # negative -> left hand, positive -> right hand.
    return HandLandmarks.LEFT if triple < 0.0 else HandLandmarks.RIGHT


class HandDetectorNode(Node):

    def __init__(self):
        super().__init__('hand_detector_node')

        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter(
            'depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter(
            'camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('num_hands', 2)
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('min_presence_confidence', 0.5)
        self.declare_parameter('trt_engine_cache_dir', DEFAULT_TRT_CACHE_DIR)

        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        num_hands = self.get_parameter('num_hands').value
        engine_cache_dir = self.get_parameter('trt_engine_cache_dir').value

        os.makedirs(engine_cache_dir, exist_ok=True)

        models_dir = os.path.join(
            get_package_share_directory('hand_detector'), 'models')

        # GPU acceleration is the default here: onnxruntime tries
        # TensorrtExecutionProvider first, then CUDAExecutionProvider, and
        # only falls back to CPU if neither is available. The first launch
        # after a cache directory is wiped will block here for several
        # minutes while TensorRT builds and caches the engines; every
        # subsequent launch reuses the cached '.engine' files and starts in
        # a few seconds.
        self.get_logger().info(
            f'Loading palm/hand-landmark ONNX models (TensorRT engine cache: '
            f'{engine_cache_dir}). First run after a cache wipe can take '
            f'several minutes to build the TensorRT engines.')
        self._palm_detection = PalmDetection(
            model_path=os.path.join(
                models_dir, 'palm_detection_full_inf_post_192x192.onnx'),
            score_threshold=self.get_parameter(
                'min_detection_confidence').value,
            max_hands=num_hands,
            engine_cache_dir=engine_cache_dir,
        )
        self._hand_landmark = HandLandmark(
            model_path=os.path.join(
                models_dir, 'hand_landmark_sparse_Nx3x224x224.onnx'),
            class_score_th=self.get_parameter(
                'min_presence_confidence').value,
            max_num_hands=num_hands,
            engine_cache_dir=engine_cache_dir,
        )
        self.get_logger().info(
            f'palm_detection providers={self._palm_detection.providers} '
            f'hand_landmark providers={self._hand_landmark.providers}')

        self._bridge = CvBridge()
        self._intrinsics = None  # (fx, fy, cx, cy)
        self._wh_ratio = None

        self._info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_cb,
            qos_profile_sensor_data)

        self._color_sub = message_filters.Subscriber(
            self, Image, color_topic, qos_profile=qos_profile_sensor_data)
        self._depth_sub = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub], queue_size=10, slop=0.05)
        self._sync.registerCallback(self._synced_cb)

        self._overlay_pub = self.create_publisher(
            Image, '/hand_detector/image_overlay', 10)
        self._landmarks_pub = self.create_publisher(
            HandLandmarksArray, '/hand_detector/hand_landmarks', 10)

        self.get_logger().info(
            f'hand_detector_node ready. color={color_topic} depth={depth_topic} '
            f'camera_info={camera_info_topic} models_dir={models_dir}')

    def _camera_info_cb(self, msg: CameraInfo):
        self._intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
        self._camera_frame_id = msg.header.frame_id

    def _synced_cb(self, color_msg: Image, depth_msg: Image):
        if self._intrinsics is None:
            self.get_logger().warn(
                'Waiting for camera_info before publishing 3D landmarks',
                throttle_duration_sec=5.0)
            return

        fx, fy, cx, cy = self._intrinsics

        # Native RealSense color stream is RGB8, so requesting rgb8 here
        # avoids an extra BGR<->RGB conversion pass over the full frame.
        rgb = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        depth = self._bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='passthrough')

        if depth_msg.encoding == '16UC1':
            depth_scale = 0.001  # millimeters -> meters
        elif depth_msg.encoding == '32FC1':
            depth_scale = 1.0  # already meters
        else:
            self.get_logger().warn(
                f'Unexpected depth encoding {depth_msg.encoding}, assuming meters',
                throttle_duration_sec=5.0)
            depth_scale = 1.0

        height, width = rgb.shape[:2]
        if self._wh_ratio is None:
            self._wh_ratio = width / height

        # Draw directly on rgb; nothing downstream still needs the
        # untouched frame, so a defensive copy would just be wasted work.
        overlay = rgb

        landmarks_array = HandLandmarksArray()
        landmarks_array.header = color_msg.header

        hands = self._palm_detection(rgb)
        # hand: sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y

        if len(hands) > 0:
            rects = []
            for sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y in hands:
                rcx = int(sqn_rr_center_x * width)
                rcy = int(sqn_rr_center_y * height)
                xmin = max(0, int((sqn_rr_center_x - sqn_rr_size / 2) * width))
                xmax = min(width, int((sqn_rr_center_x + sqn_rr_size / 2) * width))
                ymin = max(0, int((sqn_rr_center_y - sqn_rr_size * self._wh_ratio / 2) * height))
                ymax = min(height, int((sqn_rr_center_y + sqn_rr_size * self._wh_ratio / 2) * height))
                degree = np.degrees(rotation)
                rects.append([rcx, rcy, xmax - xmin, ymax - ymin, degree])
            rects = np.asarray(rects, dtype=np.float32)

            cropped_hand_images = rotate_and_crop_rectangle(
                image=rgb, rects_tmp=rects,
                operation_when_cropping_out_of_range='padding')

            if len(cropped_hand_images) > 0:
                hand_landmarks, rotated_image_size_leftrights = self._hand_landmark(
                    images=cropped_hand_images, rects=rects)

                for landmarks_21x2, size_leftright in zip(
                        hand_landmarks, rotated_image_size_leftrights):
                    pixel_coords = [(int(u), int(v)) for u, v in landmarks_21x2]

                    hand_msg = HandLandmarks()

                    # `landmarks` is a fixed-size Point[21] that rosidl has
                    # already pre-populated, so fill the existing entries in
                    # place. Appending instead would leave those 21 zeroed
                    # placeholders as the published payload and push the real
                    # coordinates out to indices 21..41, where they are
                    # dropped on serialization.
                    for point, (u, v) in zip(hand_msg.landmarks, pixel_coords):
                        if 0 <= u < width and 0 <= v < height:
                            z = float(depth[v, u]) * depth_scale
                        else:
                            z = 0.0

                        if z > 0.0:
                            point.x = (u - cx) * z / fx
                            point.y = (v - cy) * z / fy
                            point.z = z
                        else:
                            point.x = float('nan')
                            point.y = float('nan')
                            point.z = float('nan')

                    # The ONNX model's own left/right classification looks
                    # only at the 2D crop's appearance, which is ambiguous
                    # for the back of a hand - it looks like the mirror
                    # image (i.e. the OTHER hand's palm), so back-facing
                    # hands get misclassified. With real depth we have true
                    # 3D landmark positions, so compute chirality
                    # geometrically instead - that's invariant to viewing
                    # angle (front/back), only flipping for an actual
                    # mirror-image (left vs right) hand. Falls back to the
                    # model's classification if depth was invalid at the
                    # landmarks it needs.
                    handedness_3d = _handedness_from_3d(hand_msg.landmarks)
                    if handedness_3d is not None:
                        hand_msg.handedness = handedness_3d
                    else:
                        hand_msg.handedness = (
                            HandLandmarks.RIGHT if size_leftright[2] >= 0.5
                            else HandLandmarks.LEFT)

                    landmarks_array.hands.append(hand_msg)

                    for connection in HAND_CONNECTIONS:
                        p1 = pixel_coords[connection[0]]
                        p2 = pixel_coords[connection[1]]
                        cv2.line(overlay, p1, p2, (0, 255, 0), 2)
                    for (u, v) in pixel_coords:
                        cv2.circle(overlay, (u, v), 4, (255, 0, 0), -1)

        if len(landmarks_array.hands) > 0:
            self._landmarks_pub.publish(landmarks_array)

        overlay_msg = Image()
        overlay_msg.header = color_msg.header
        overlay_msg.height, overlay_msg.width = overlay.shape[:2]
        overlay_msg.encoding = 'rgb8'
        overlay_msg.is_bigendian = 0
        overlay_msg.step = overlay.shape[1] * 3
        # Assigning `bytes` here would route through Image.data's fset,
        # which — unless Python runs with assertions stripped (-O) —
        # validates every single byte in a Python-level loop (~190ms for a
        # 640x480 frame). array.array hits an early-return fast path in
        # that same setter that skips the per-element loop entirely.
        overlay_msg.data = array.array(
            'B', np.ascontiguousarray(overlay).tobytes())
        self._overlay_pub.publish(overlay_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HandDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # SIGINT delivered to a spinning executor surfaces as
        # ExternalShutdownException, not KeyboardInterrupt - rclpy has
        # already called context.shutdown() by the time this is raised.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        # The onnxruntime-gpu wheel used here (Jetson community build)
        # double-frees a native allocation somewhere in its pybind11
        # extension's teardown path, which CPython triggers while
        # clearing module globals during normal interpreter shutdown
        # (Py_FinalizeEx -> _PyModule_ClearDict). That is a bug inside
        # the prebuilt onnxruntime binary, not in this process's own
        # state, and it is unrelated to whether GPU inference actually
        # ran. os._exit() skips CPython's module teardown entirely, so
        # it never reaches the broken destructor; the OS reclaims all
        # process resources regardless.
        os._exit(0)


if __name__ == '__main__':
    main()
