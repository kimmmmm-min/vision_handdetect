# Filtered 3D motion state of a tracked hand (constant-acceleration Kalman filter output)

std_msgs/Header header

uint8 LEFT=0
uint8 RIGHT=1
uint8 handedness

# true when this cycle had a measurement accepted by the filter;
# false when coasting on prediction alone (hand not detected / rejected outlier)
bool tracking

geometry_msgs/Point position          # meters, camera optical frame
geometry_msgs/Vector3 velocity        # m/s
geometry_msgs/Vector3 acceleration    # m/s^2

float64 distance                # d = |position|, meters
float64 distance_rate           # d_dot, m/s
float64 distance_acceleration   # d_ddot, m/s^2
