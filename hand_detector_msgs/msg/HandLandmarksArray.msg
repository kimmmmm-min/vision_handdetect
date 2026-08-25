# Per-frame collection of detected hands, each independently labeled
# left/right and ordered arbitrarily (order is not stable across frames -
# use `handedness` to associate a hand across messages, not array index).

std_msgs/Header header
HandLandmarks[] hands
