"""Hand landmark regressor (BlazeHand) inference via onnxruntime.

Like PalmDetection, GPU acceleration (TensorRT -> CUDA -> CPU) is the
default. The model has a dynamic batch dimension ('N' hands per frame), so
the TensorRT execution provider needs an explicit optimization profile
(min/opt/max shapes) - without it, TensorRT engine construction fails
immediately. min=1 covers the "no hand" call site never happening (callers
skip inference when there are zero crops); max=`max_num_hands` covers the
worst case of every supported hand being visible at once.
"""
import copy
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime

from hand_detector.onnx_hand.utils import keep_aspect_resize_and_pad


def _default_providers(max_num_hands: int, engine_cache_dir: Optional[str]):
    trt_options = {
        'trt_engine_cache_enable': True,
        'trt_fp16_enable': True,
        'trt_profile_min_shapes': f'input:1x3x224x224',
        'trt_profile_opt_shapes': f'input:{max_num_hands}x3x224x224',
        'trt_profile_max_shapes': f'input:{max_num_hands}x3x224x224',
    }
    if engine_cache_dir is not None:
        trt_options['trt_engine_cache_path'] = engine_cache_dir
    return [
        ('TensorrtExecutionProvider', trt_options),
        'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ]


class HandLandmark:

    def __init__(
        self,
        model_path: str,
        class_score_th: float = 0.50,
        max_num_hands: int = 2,
        engine_cache_dir: Optional[str] = None,
        providers: Optional[List] = None,
    ):
        self.class_score_th = class_score_th

        providers = providers if providers is not None else _default_providers(
            max_num_hands, engine_cache_dir,
        )

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

    def __call__(
        self,
        images: List[np.ndarray],
        rects: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        images: cropped+derotated palm images (one per detected hand)
        rects: [N, cx, cy, width, height, degree] (same as PalmDetection's rects)

        Returns
        -------
        hand_landmarks: int32[N, 21, 2] pixel (x, y) in the ORIGINAL image
        rotated_image_size_leftrights: [N, 3] = rotated_w, rotated_h, 0(left)/1(right)
        """
        temp_images = copy.deepcopy(images)
        inference_images, resized_images, resize_scales, half_pad_sizes = self.__preprocess(temp_images)

        xyz_x21s, hand_scores, left_hand_0_or_right_hand_1s = self.onnx_session.run(
            self.output_names,
            {name: inference_images for name in self.input_names},
        )

        return self.__postprocess(
            resized_images=resized_images,
            resize_scales_224x224=resize_scales,
            half_pad_sizes_224x224=half_pad_sizes,
            rects=rects,
            xyz_x21s=xyz_x21s,
            hand_scores=hand_scores,
            left_hand_0_or_right_hand_1s=left_hand_0_or_right_hand_1s,
        )

    def __preprocess(
        self,
        images: List[np.ndarray],
        swap: Tuple[int, int, int] = (2, 0, 1),
    ):
        input_h = self.input_shapes[0][2]
        input_w = self.input_shapes[0][3]

        padded_images, resized_images = [], []
        resize_scales_224x224, half_pad_sizes_224x224 = [], []

        for image in images:
            padded_image, resized_image = keep_aspect_resize_and_pad(
                image=image, resize_width=input_w, resize_height=input_h,
            )
            resize_scale_h = resized_image.shape[0] / image.shape[0]
            resize_scale_w = resized_image.shape[1] / image.shape[1]
            resize_scales_224x224.append([resize_scale_w, resize_scale_h])

            pad_h = padded_image.shape[0] - resized_image.shape[0]
            pad_w = padded_image.shape[1] - resized_image.shape[1]
            half_pad_sizes_224x224.append([max(0, pad_w // 2), max(0, pad_h // 2)])

            padded_image = np.divide(padded_image, 255.0)
            # No channel flip: crops come from the already-RGB frame
            # (see palm_detection.py's __preprocess for why).
            padded_image = padded_image.transpose(swap)
            padded_images.append(np.ascontiguousarray(padded_image, dtype=np.float32))
            resized_images.append(resized_image)

        return (
            np.asarray(padded_images, dtype=np.float32),
            resized_images,
            np.asarray(resize_scales_224x224, dtype=np.float32),
            np.asarray(half_pad_sizes_224x224, dtype=np.int32),
        )

    def __postprocess(
        self,
        resized_images: List[np.ndarray],
        resize_scales_224x224: np.ndarray,
        half_pad_sizes_224x224: np.ndarray,
        rects: np.ndarray,
        xyz_x21s: np.ndarray,
        hand_scores: np.ndarray,
        left_hand_0_or_right_hand_1s: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        extracted_hands = []
        rotated_image_size_leftrights = []

        keep = hand_scores[:, 0] > self.class_score_th
        xyz_x21s = xyz_x21s[keep, :]
        hand_scores = hand_scores[keep, :]
        left_hand_0_or_right_hand_1s = left_hand_0_or_right_hand_1s[keep, :]
        resize_scales_224x224 = resize_scales_224x224[keep, :]
        half_pad_sizes_224x224 = half_pad_sizes_224x224[keep, :]
        rects = rects[keep, :]
        resized_images = [img for img, k in zip(resized_images, keep) if k]

        input_h = self.input_shapes[0][2]
        input_w = self.input_shapes[0][3]

        for resized_image, resize_scale, half_pad_size, rect, xyz_x21, left_or_right in zip(
            resized_images, resize_scales_224x224, half_pad_sizes_224x224,
            rects, xyz_x21s, left_hand_0_or_right_hand_1s,
        ):
            rrn_lms = xyz_x21 / input_h
            rcx, rcy, angle = rect[0], rect[1], rect[4]

            view_image = copy.deepcopy(resized_image)
            view_image = cv2.resize(
                view_image, dsize=None, fx=1 / resize_scale[0], fy=1 / resize_scale[1],
            )
            rescaled_xy = np.asarray(
                [[v[0], v[1]] for v in zip(rrn_lms[0::3], rrn_lms[1::3])], dtype=np.float32,
            )
            rescaled_xy[:, 0] = (rescaled_xy[:, 0] * input_w - half_pad_size[0]) / resize_scale[0]
            rescaled_xy[:, 1] = (rescaled_xy[:, 1] * input_h - half_pad_size[1]) / resize_scale[1]
            rescaled_xy = rescaled_xy.astype(np.int32)

            height, width = view_image.shape[:2]
            image_center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(image_center, -int(angle), 1)
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)
            rotation_matrix[0, 2] += bound_w / 2 - image_center[0]
            rotation_matrix[1, 2] += bound_h / 2 - image_center[1]

            keypoints = []
            for x, y in rescaled_xy:
                coord = np.array([[x, y, 1]])
                new_coord = rotation_matrix.dot(coord.T)
                keypoints.append([int(new_coord[0]), int(new_coord[1])])

            rotated_w, rotated_h = bound_w, bound_h
            half_w, half_h = rotated_w // 2, rotated_h // 2

            hand_landmarks = np.asarray(keypoints, dtype=np.int32).reshape(-1, 2)
            hand_landmarks[..., 0] = hand_landmarks[..., 0] + rcx - half_w
            hand_landmarks[..., 1] = hand_landmarks[..., 1] + rcy - half_h
            extracted_hands.append(hand_landmarks)
            rotated_image_size_leftrights.append([rotated_w, rotated_h, float(left_or_right)])

        return (
            np.asarray(extracted_hands, dtype=np.int32),
            np.asarray(rotated_image_size_leftrights),
        )
