# 3D landmarks for a single detected hand, with per-hand handedness.
# handedness matches the ONNX hand-landmark model's own left/right
# classification (camera's point of view, no image flip in this pipeline).

uint8 LEFT=0
uint8 RIGHT=1

uint8 handedness
geometry_msgs/Point[21] landmarks
