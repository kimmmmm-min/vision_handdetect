"""Debug visualization: draws a velocity/acceleration arrow for each tracked
hand onto Stage 1's skeleton overlay, using Stage 2's Kalman-filtered
HandMotion output.

Kept as its own node (Stage 3) rather than folded into hand_motion_node so
Stage 2 stays a lightweight CPU-only Kalman filter with no image/cv_bridge
dependency - this is purely an opt-in debugging aid.

Sync strategy: rather than matching each image to the HandMotion(s) that
share its exact stamp (which meant re-publishing the same image once per
image arrival and again per late-arriving hand motion - up to 3 publishes
for one logical camera frame, which reads as the arrow blinking in and out
on a rendered stream), each incoming image is drawn with whatever is the
*latest known* HandMotion per hand (LEFT/RIGHT independently) as long as
it isn't older than `motion_staleness_sec`, and published exactly once.
/hand_detector/image_overlay and /hand_detector/hand_motion both carry the
header.stamp of the original color frame they derived from
(hand_detector_node.py sets both from the same color_msg.header, and that
stamp survives unchanged through hand_motion_node.py), so "how old" a
motion is relative to the current image is measured in that shared camera
clock rather than wall-clock arrival time.
"""
import array
import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

from hand_detector_msgs.msg import HandMotion

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (160, 160, 160)

ARROWHEAD_LEN_PX = 14
ARROWHEAD_ANGLE_DEG = 28
DASH_LEN_PX = 9
DASH_GAP_PX = 6
HOLLOW_OUTLINE_PX = 3
DOT_RADIUS_PX = 5
# horizontal arrow with its tip drooping downward by this much, so it
# reads as "roughly sideways" rather than a perfectly flat line.
ARROW_DROOP_DEG = 30.0


def stamp_to_seconds(stamp) -> float:
    return Time.from_msg(stamp).nanoseconds * 1e-9


class HandMotionOverlayNode(Node):

    def __init__(self):
        super().__init__('hand_motion_overlay_node')

        self.declare_parameter('image_topic', '/hand_detector/image_overlay')
        self.declare_parameter('motion_topic', '/hand_detector/hand_motion')
        self.declare_parameter(
            'camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('output_topic', '/hand_detector/motion_overlay')

        # |distance_rate| at or below this -> gray + dot (no reliable
        # toward/away direction to show); above -> red arrow up
        # (approaching) / blue arrow down (receding).
        self.declare_parameter('d_dot_threshold', 0.02)  # m/s
        self.declare_parameter('arrow_length_scale', 150.0)  # px per (m/s)
        self.declare_parameter('arrow_min_length_px', 10.0)
        self.declare_parameter('arrow_max_length_px', 220.0)
        self.declare_parameter('arrow_thin_thickness', 3)
        self.declare_parameter('arrow_thick_thickness', 12)
        # a hand's most recent HandMotion is reused on every new image frame
        # until it's older than this, rather than only drawing on the exact
        # stamp it arrived with - keeps exactly one publish per image
        # instead of re-publishing once per image and again per hand motion
        # (which was making the arrow blink in/out on a rendered stream).
        self.declare_parameter('motion_staleness_sec', 0.2)

        image_topic = self.get_parameter('image_topic').value
        motion_topic = self.get_parameter('motion_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        output_topic = self.get_parameter('output_topic').value

        self._bridge = CvBridge()
        self._intrinsics = None  # (fx, fy, cx, cy)

        # handedness -> most recently received HandMotion for that hand.
        self._latest_motion = {}

        self._info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self._info_cb,
            qos_profile_sensor_data)
        # image_overlay/hand_motion are published with the default reliable
        # QoS (depth 10), not qos_profile_sensor_data - must match or no
        # messages are ever delivered.
        self._image_sub = self.create_subscription(
            Image, image_topic, self._image_cb, 10)
        self._motion_sub = self.create_subscription(
            HandMotion, motion_topic, self._motion_cb, 10)

        self._pub = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info(
            f'hand_motion_overlay_node ready. image={image_topic} '
            f'motion={motion_topic} camera_info={camera_info_topic} '
            f'output={output_topic}')

    def _info_cb(self, msg: CameraInfo):
        self._intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _image_cb(self, msg: Image):
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8').copy()

        if self._intrinsics is not None:
            now_s = stamp_to_seconds(msg.header.stamp)
            staleness = self.get_parameter('motion_staleness_sec').value
            for motion in self._latest_motion.values():
                motion_s = stamp_to_seconds(motion.header.stamp)
                if now_s - motion_s <= staleness:
                    self._draw_motion(img, motion)

        self._publish(img, msg.header)

    def _motion_cb(self, msg: HandMotion):
        self._latest_motion[msg.handedness] = msg

    def _draw_motion(self, img: np.ndarray, motion: HandMotion):
        fx, fy, cx, cy = self._intrinsics
        x, y, z = motion.position.x, motion.position.y, motion.position.z
        if z <= 0.0 or math.isnan(x) or math.isnan(y) or math.isnan(z):
            return

        u0 = fx * x / z + cx
        v0 = fy * y / z + cy
        height, width = img.shape[:2]
        if not (0 <= u0 < width and 0 <= v0 < height):
            return

        # Direction is fixed horizontal (right = approaching, left =
        # receding), driven by distance_rate's sign rather than the raw
        # (vx, vy) components: atan2(vy, vx) is extremely noise-sensitive
        # whenever lateral speed is small, which was making the arrow's
        # heading jitter frame to frame. distance_rate is far more stable,
        # and a fixed direction also makes length/thickness easier to
        # compare by eye across frames (nothing to mentally rotate).
        droop = math.radians(ARROW_DROOP_DEG)
        d_dot_threshold = self.get_parameter('d_dot_threshold').value
        if motion.distance_rate < -d_dot_threshold:
            color = RED            # approaching
            direction = (math.cos(droop), math.sin(droop))   # right, tip drooping down
        elif motion.distance_rate > d_dot_threshold:
            color = BLUE           # receding
            direction = (-math.cos(droop), math.sin(droop))  # left, tip drooping down
        else:
            color = GRAY      # near-stationary along line of sight
            direction = None  # no meaningful toward/away direction to show

        vx, vy, vz = motion.velocity.x, motion.velocity.y, motion.velocity.z
        speed3d = math.sqrt(vx * vx + vy * vy + vz * vz)

        ax, ay, az = motion.acceleration.x, motion.acceleration.y, motion.acceleration.z
        # sign(V.A) == sign(d|V|/dt): whether 3D speed is increasing
        # (accelerating -> thick) or decreasing (decelerating -> thin).
        speeding_up = (vx * ax + vy * ay + vz * az) > 0.0
        thickness = (
            self.get_parameter('arrow_thick_thickness').value if speeding_up
            else self.get_parameter('arrow_thin_thickness').value)

        # tracking=False means this cycle had no accepted measurement -
        # the filter is coasting on prediction alone. Still draw it (that
        # coasting behavior is exactly what's useful to see while
        # debugging), but visually mark it as unconfirmed: dashed shaft,
        # hollow arrowhead/dot instead of solid/filled.
        predicted = not motion.tracking

        if direction is None:
            self._draw_dot(img, (u0, v0), color, filled=not predicted)
        else:
            length_px = float(np.clip(
                self.get_parameter('arrow_length_scale').value * speed3d,
                self.get_parameter('arrow_min_length_px').value,
                self.get_parameter('arrow_max_length_px').value))
            p0 = (u0, v0)
            p1 = (u0 + direction[0] * length_px, v0 + direction[1] * length_px)
            self._draw_arrow(img, p0, p1, color, thickness, dashed=predicted,
                             hollow_head=predicted)

    def _draw_dot(self, img, center, color, filled: bool):
        c = (int(round(center[0])), int(round(center[1])))
        cv2.circle(img, c, DOT_RADIUS_PX, color,
                  -1 if filled else 1, cv2.LINE_AA)

    def _draw_arrow(self, img, p0, p1, color, thickness, dashed: bool,
                    hollow_head: bool):
        p0i = (int(round(p0[0])), int(round(p0[1])))
        p1i = (int(round(p1[0])), int(round(p1[1])))

        # Stop the shaft short of the tip, at the base of the arrowhead -
        # otherwise the shaft is drawn the full distance underneath the
        # head, and for a HOLLOW head that shaft (dashed or not) shows
        # right through the "empty" interior instead of leaving it clear.
        head_len = max(ARROWHEAD_LEN_PX, thickness * 2.2)
        dist = math.hypot(p1i[0] - p0i[0], p1i[1] - p0i[1])
        if dist > head_len:
            frac = (dist - head_len) / dist
            shaft_end = (p0i[0] + (p1i[0] - p0i[0]) * frac,
                         p0i[1] + (p1i[1] - p0i[1]) * frac)
            shaft_end = (int(round(shaft_end[0])), int(round(shaft_end[1])))
        else:
            shaft_end = p0i  # arrow too short for a separate shaft segment

        if dashed:
            self._draw_dashed_line(img, p0i, shaft_end, color, thickness)
        else:
            cv2.line(img, p0i, shaft_end, color, thickness, cv2.LINE_AA)
        self._draw_arrowhead(img, p0i, p1i, color, thickness,
                             filled=not hollow_head)

    def _draw_dashed_line(self, img, p0, p1, color, thickness):
        x0, y0 = p0
        x1, y1 = p1
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1e-3:
            return
        ux, uy = (x1 - x0) / dist, (y1 - y0) / dist
        # cv2.line draws thick strokes with round end caps, whose radius
        # (~thickness/2 on each side) can visually bridge a gap that's
        # narrower than the shaft is thick - the dashes then read as one
        # solid line. Padding the gap by the shaft thickness keeps an
        # actual visible break regardless of how bold the line is.
        effective_gap = DASH_GAP_PX + thickness
        pos, draw = 0.0, True
        while pos < dist:
            seg = DASH_LEN_PX if draw else effective_gap
            end = min(pos + seg, dist)
            if draw:
                sx, sy = x0 + ux * pos, y0 + uy * pos
                ex, ey = x0 + ux * end, y0 + uy * end
                cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)),
                         color, thickness, cv2.LINE_AA)
            pos, draw = end, not draw

    def _draw_arrowhead(self, img, p0, p1, color, thickness, filled: bool):
        x0, y0 = p0
        x1, y1 = p1
        if x0 == x1 and y0 == y1:
            return
        # scale the head with shaft thickness so a thick (accelerating)
        # arrow doesn't end up with a disproportionately small tip.
        head_len = max(ARROWHEAD_LEN_PX, thickness * 2.2)
        theta = math.atan2(y1 - y0, x1 - x0)
        a = math.radians(ARROWHEAD_ANGLE_DEG)
        back1 = (x1 - head_len * math.cos(theta - a),
                 y1 - head_len * math.sin(theta - a))
        back2 = (x1 - head_len * math.cos(theta + a),
                 y1 - head_len * math.sin(theta + a))
        pts = np.array([p1, back1, back2], dtype=np.int32)
        if filled:
            cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA)
        else:
            # a thick outline on a small triangle just fills it back in
            # (defeating the point of "hollow") - use a slim, FIXED stroke
            # here, independent of the shaft's accel/decel thickness.
            cv2.polylines(img, [pts], isClosed=True, color=color,
                          thickness=HOLLOW_OUTLINE_PX, lineType=cv2.LINE_AA)

    def _publish(self, img: np.ndarray, header):
        msg = Image()
        msg.header = header
        msg.height, msg.width = img.shape[:2]
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = img.shape[1] * 3
        # Same array.array fast-path as hand_detector_node.py's overlay
        # publish - see that file for why raw `bytes` is ~190ms slower.
        msg.data = array.array('B', np.ascontiguousarray(img).tobytes())
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HandMotionOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
