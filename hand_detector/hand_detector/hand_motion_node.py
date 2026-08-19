import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseArray

from hand_detector_msgs.msg import HandMotion

NUM_LANDMARKS = 21
# indices averaged to get the palm-center representative point
PALM_LANDMARK_INDICES = (0, 5, 17)

# chi-square critical value, 3 degrees of freedom, 99% confidence.
# Used to gate measurements that are statistically implausible given the
# filter's current predicted state + covariance (rejects depth outliers).
CHI2_GATE_3DOF_99 = 11.345


def stamp_to_seconds(stamp) -> float:
    return Time.from_msg(stamp).nanoseconds * 1e-9


class ConstantAccelerationKalmanFilter:
    """3D constant-acceleration Kalman filter.

    State: [X, Y, Z, VX, VY, VZ, AX, AY, AZ]
    Measurement: [X, Y, Z]
    """

    STATE_DIM = 9
    MEAS_DIM = 3

    def __init__(self, process_noise_sigma: float,
                 initial_pos_var: float, initial_vel_var: float,
                 initial_accel_var: float):
        self.sigma_a = process_noise_sigma
        self._initial_pos_var = initial_pos_var
        self._initial_vel_var = initial_vel_var
        self._initial_accel_var = initial_accel_var

        self.x = np.zeros(self.STATE_DIM)
        self.P = np.eye(self.STATE_DIM)

        self.H = np.zeros((self.MEAS_DIM, self.STATE_DIM))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

    def reinit(self, position: np.ndarray):
        self.x = np.zeros(self.STATE_DIM)
        self.x[0:3] = position
        self.P = np.diag([
            self._initial_pos_var, self._initial_pos_var, self._initial_pos_var,
            self._initial_vel_var, self._initial_vel_var, self._initial_vel_var,
            self._initial_accel_var, self._initial_accel_var, self._initial_accel_var,
        ])

    def _transition_matrix(self, dt: float) -> np.ndarray:
        F = np.eye(self.STATE_DIM)
        half_dt2 = 0.5 * dt * dt
        for pos_idx, vel_idx, acc_idx in ((0, 3, 6), (1, 4, 7), (2, 5, 8)):
            F[pos_idx, vel_idx] = dt
            F[pos_idx, acc_idx] = half_dt2
            F[vel_idx, acc_idx] = dt
        return F

    def _process_noise(self, dt: float) -> np.ndarray:
        # Discretized "constant Wiener process acceleration" noise model
        # (white-noise jerk of variance sigma_a^2) for one axis:
        #   [[dt^5/20, dt^4/8,  dt^3/6],
        #    [dt^4/8,  dt^3/3,  dt^2/2],
        #    [dt^3/6,  dt^2/2,  dt    ]] * sigma_a^2
        s2 = self.sigma_a ** 2
        dt2, dt3, dt4, dt5 = dt ** 2, dt ** 3, dt ** 4, dt ** 5
        q_pp, q_pv, q_pa = dt5 / 20.0, dt4 / 8.0, dt3 / 6.0
        q_vv, q_va = dt3 / 3.0, dt2 / 2.0
        q_aa = dt

        Q = np.zeros((self.STATE_DIM, self.STATE_DIM))
        for pos_idx, vel_idx, acc_idx in ((0, 3, 6), (1, 4, 7), (2, 5, 8)):
            Q[pos_idx, pos_idx] = q_pp
            Q[pos_idx, vel_idx] = Q[vel_idx, pos_idx] = q_pv
            Q[pos_idx, acc_idx] = Q[acc_idx, pos_idx] = q_pa
            Q[vel_idx, vel_idx] = q_vv
            Q[vel_idx, acc_idx] = Q[acc_idx, vel_idx] = q_va
            Q[acc_idx, acc_idx] = q_aa
        return Q * s2

    def predict(self, dt: float):
        F = self._transition_matrix(dt)
        Q = self._process_noise(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z: np.ndarray, R: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        return y, S

    def mahalanobis_sq(self, y: np.ndarray, S: np.ndarray) -> float:
        return float(y.T @ np.linalg.inv(S) @ y)

    def update(self, z: np.ndarray, R: np.ndarray):
        y, S = self.innovation(z, R)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.STATE_DIM)
        self.P = (I - K @ self.H) @ self.P


class HandMotionNode(Node):

    def __init__(self):
        super().__init__('hand_motion_node')

        self.declare_parameter('landmarks_topic', '/hand_detector/hand_landmarks')
        self.declare_parameter('motion_topic', '/hand_detector/hand_motion')
        self.declare_parameter('process_noise_sigma', 3.0)  # m/s^3 (jerk std)
        self.declare_parameter('depth_relative_error', 0.02)  # 2% at any range
        self.declare_parameter('min_measurement_sigma', 0.005)  # meters
        self.declare_parameter('max_range', 5.0)  # meters, sanity bound
        self.declare_parameter('reset_timeout_sec', 0.5)
        self.declare_parameter('gate_chi2_threshold', CHI2_GATE_3DOF_99)
        self.declare_parameter('initial_pos_var', 0.01)
        self.declare_parameter('initial_vel_var', 1.0)
        self.declare_parameter('initial_accel_var', 4.0)

        landmarks_topic = self.get_parameter('landmarks_topic').value
        motion_topic = self.get_parameter('motion_topic').value
        self._depth_rel_error = self.get_parameter('depth_relative_error').value
        self._min_meas_sigma = self.get_parameter('min_measurement_sigma').value
        self._max_range = self.get_parameter('max_range').value
        self._reset_timeout = self.get_parameter('reset_timeout_sec').value
        self._gate_threshold = self.get_parameter('gate_chi2_threshold').value

        self._kf = ConstantAccelerationKalmanFilter(
            process_noise_sigma=self.get_parameter('process_noise_sigma').value,
            initial_pos_var=self.get_parameter('initial_pos_var').value,
            initial_vel_var=self.get_parameter('initial_vel_var').value,
            initial_accel_var=self.get_parameter('initial_accel_var').value,
        )

        self._initialized = False
        self._last_proc_time_s = None
        self._last_meas_time_s = None

        self._sub = self.create_subscription(
            PoseArray, landmarks_topic, self._landmarks_cb, 10)
        self._pub = self.create_publisher(HandMotion, motion_topic, 10)

        self.get_logger().info(
            f'hand_motion_node ready. landmarks={landmarks_topic} motion={motion_topic}')

    def _extract_palm_center(self, msg: PoseArray):
        if len(msg.poses) < NUM_LANDMARKS:
            return None

        pts = []
        for idx in PALM_LANDMARK_INDICES:
            p = msg.poses[idx].position
            if math.isnan(p.x) or math.isnan(p.y) or math.isnan(p.z):
                return None
            pts.append((p.x, p.y, p.z))

        center = np.mean(np.array(pts), axis=0)

        if center[2] <= 0.0 or np.linalg.norm(center) > self._max_range:
            return None
        if np.allclose(center, 0.0):
            return None

        return center

    def _measurement_noise(self, measured_distance: float) -> np.ndarray:
        sigma = max(self._depth_rel_error * measured_distance, self._min_meas_sigma)
        return np.diag([sigma ** 2, sigma ** 2, sigma ** 2])

    def _landmarks_cb(self, msg: PoseArray):
        now_s = stamp_to_seconds(msg.header.stamp)
        measurement = self._extract_palm_center(msg)

        if not self._initialized:
            if measurement is None:
                return
            self._kf.reinit(measurement)
            self._initialized = True
            self._last_proc_time_s = now_s
            self._last_meas_time_s = now_s
            self._publish(msg.header, tracking=True)
            return

        dt = now_s - self._last_proc_time_s
        if dt <= 0.0:
            self.get_logger().warn(
                'Non-increasing timestamp on hand_landmarks, dropping message',
                throttle_duration_sec=5.0)
            return
        self._last_proc_time_s = now_s

        self._kf.predict(dt)

        accepted = False
        if measurement is not None:
            distance = float(np.linalg.norm(measurement))
            R = self._measurement_noise(distance)
            y, S = self._kf.innovation(measurement, R)
            if self._kf.mahalanobis_sq(y, S) <= self._gate_threshold:
                self._kf.update(measurement, R)
                accepted = True
                self._last_meas_time_s = now_s

        if not accepted:
            since_last_meas = now_s - self._last_meas_time_s
            if since_last_meas > self._reset_timeout:
                self.get_logger().warn(
                    f'No accepted hand measurement for {since_last_meas:.2f}s, '
                    'resetting motion filter', throttle_duration_sec=1.0)
                self._initialized = False
                return

        self._publish(msg.header, tracking=accepted)

    def _publish(self, header, tracking: bool):
        x = self._kf.x
        position = x[0:3]
        velocity = x[3:6]
        acceleration = x[6:9]

        distance = float(np.linalg.norm(position))
        if distance > 1e-6:
            distance_rate = float(np.dot(position, velocity) / distance)
            speed_sq = float(np.dot(velocity, velocity))
            distance_accel = float(
                (speed_sq + np.dot(position, acceleration) - distance_rate ** 2)
                / distance)
        else:
            distance_rate = 0.0
            distance_accel = 0.0

        out = HandMotion()
        out.header = header
        out.tracking = tracking
        out.position.x, out.position.y, out.position.z = position.tolist()
        out.velocity.x, out.velocity.y, out.velocity.z = velocity.tolist()
        out.acceleration.x, out.acceleration.y, out.acceleration.z = (
            acceleration.tolist())
        out.distance = distance
        out.distance_rate = distance_rate
        out.distance_acceleration = distance_accel
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HandMotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
