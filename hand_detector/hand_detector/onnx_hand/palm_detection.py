"""Palm detector (BlazePalm) inference via onnxruntime.

GPU acceleration is the default: onnxruntime is asked for
TensorrtExecutionProvider first, then CUDAExecutionProvider, then CPU as a
last-resort fallback. The TensorRT engine is cached on disk so only the very
first run pays the multi-minute engine-build cost; subsequent runs (and
subsequent process launches) load the cached '.engine' file in a few
seconds.
"""
import copy
from math import sin, cos, atan2, pi
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime

from hand_detector.onnx_hand.utils import normalize_radians, keep_aspect_resize_and_pad

DEFAULT_PROVIDERS = [
    (
        'TensorrtExecutionProvider', {
            'trt_engine_cache_enable': True,
            'trt_fp16_enable': True,
        }
    ),
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
]


class PalmDetection:

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.60,
        max_hands: int = 2,
        nms_iou_threshold: float = 0.3,
        engine_cache_dir: Optional[str] = None,
        providers: Optional[List] = None,
    ):
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        # HandLandmark's TensorRT engine is built with a fixed optimization
        # profile capped at max_hands (see hand_landmark.py's
        # trt_profile_max_shapes) - feeding it a bigger batch than that
        # isn't a graceful failure, it's a hard TensorRT EP_FAIL that
        # crashes the whole node. Capping here (keeping the
        # highest-confidence detections) is what actually enforces that
        # limit; nothing downstream does.
        self.max_hands = max_hands

        providers = providers if providers is not None else copy.deepcopy(DEFAULT_PROVIDERS)
        if engine_cache_dir is not None:
            for p in providers:
                if isinstance(p, tuple) and p[0] == 'TensorrtExecutionProvider':
                    p[1]['trt_engine_cache_path'] = engine_cache_dir

        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [i.shape for i in self.onnx_session.get_inputs()]
        self.input_names = [i.name for i in self.onnx_session.get_inputs()]
        self.output_names = [o.name for o in self.onnx_session.get_outputs()]
        self.square_standard_size = 0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Returns float32[N, 4]: sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y."""
        temp_image = copy.deepcopy(image)
        inference_image = self.__preprocess(temp_image)
        inference_image = np.asarray([inference_image], dtype=np.float32)
        boxes = self.onnx_session.run(
            self.output_names,
            {name: inference_image for name in self.input_names},
        )
        return self.__postprocess(image=temp_image, boxes=boxes[0])

    def __preprocess(
        self,
        image: np.ndarray,
        swap: Tuple[int, int, int] = (2, 0, 1),
    ) -> np.ndarray:
        input_h = self.input_shapes[0][2]
        input_w = self.input_shapes[0][3]
        image_height, image_width = image.shape[:2]

        self.square_standard_size = max(image_height, image_width)
        self.square_padding_half_size = abs(image_height - image_width) // 2

        padded_image, resized_image = keep_aspect_resize_and_pad(
            image=image, resize_width=input_w, resize_height=input_h,
        )

        pad_size_half_h = max(0, (input_h - resized_image.shape[0]) // 2)
        pad_size_half_w = max(0, (input_w - resized_image.shape[1]) // 2)
        self.pad_size_scale_h = pad_size_half_h / input_h
        self.pad_size_scale_w = pad_size_half_w / input_w

        padded_image = np.divide(padded_image, 255.0)
        # No channel flip here: the caller (hand_detector_node) already
        # hands us RGB (via cv_bridge desired_encoding='rgb8'), and the
        # model expects RGB. The upstream reference implementation this
        # was ported from reads frames with cv2.VideoCapture (BGR) and
        # flips here to get RGB - flipping our already-RGB input would
        # feed the model BGR instead, which tanks detection accuracy.
        padded_image = padded_image.transpose(swap)
        return np.ascontiguousarray(padded_image, dtype=np.float32)

    @staticmethod
    def _nms(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Greedy NMS over the raw axis-aligned (box_x, box_y, box_size)
        squares, highest score first.

        This model's raw output has no built-in deduplication: a single
        physical hand routinely trips several nearby anchors at once, each
        clearing score_threshold independently. Without merging those down
        to one detection first, capping "keep the top max_hands by score"
        can keep two overlapping boxes for the SAME hand while dropping a
        genuinely different, lower-scoring second hand.
        """
        if len(boxes) <= 1:
            return boxes

        scores = boxes[:, 0]
        half = boxes[:, 3] / 2.0
        x1, y1 = boxes[:, 1] - half, boxes[:, 2] - half
        x2, y2 = boxes[:, 1] + half, boxes[:, 2] + half
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(scores)[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            union = areas[i] + areas[rest] - inter
            iou = np.where(union > 0, inter / union, 0.0)
            order = rest[iou <= iou_threshold]

        return boxes[keep, :]

    def __postprocess(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        hands = []
        keep = boxes[:, 0] > self.score_threshold
        boxes = boxes[keep, :]

        boxes = self._nms(boxes, self.nms_iou_threshold)

        # HandLandmark's TensorRT engine is built with a fixed optimization
        # profile capped at max_hands (see hand_landmark.py's
        # trt_profile_max_shapes) - feeding it a bigger batch than that
        # isn't a graceful failure, it's a hard TensorRT EP_FAIL that
        # crashes the whole node. This is the actual enforcement of that
        # limit; nothing downstream does it. NMS above already removed
        # same-hand duplicates, so what's left here (if still over
        # max_hands) is presumably distinct detections - keep the
        # highest-confidence ones.
        if len(boxes) > self.max_hands:
            top = np.argsort(boxes[:, 0])[::-1][:self.max_hands]
            boxes = boxes[top, :]

        for box in boxes:
            pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y = box
            if box_size > 0:
                kp02_x = kp2_x - kp0_x
                kp02_y = kp2_y - kp0_y
                sqn_rr_size = 2.9 * box_size
                rotation = normalize_radians(0.5 * pi - atan2(-kp02_y, kp02_x))
                sqn_rr_center_x = box_x + 0.5 * box_size * sin(rotation)
                sqn_rr_center_y = box_y - 0.5 * box_size * cos(rotation)
                sqn_rr_center_y = (
                    sqn_rr_center_y * self.square_standard_size
                    - self.square_padding_half_size
                ) / image_height
                hands.append([sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y])

        return np.asarray(hands)
