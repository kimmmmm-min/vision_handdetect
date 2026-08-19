"""Geometry helpers for the ONNX/TensorRT palm-detection + hand-landmark pipeline.

Ported from PINTO0309/hand-gesture-recognition-using-onnx (MIT-style usage,
public reference implementation for running MediaPipe's BlazePalm /
BlazeHand models outside of the MediaPipe runtime).
"""
import copy
import math
from math import sin, cos, pi, floor
from typing import List, Tuple

import cv2
import numpy as np


def normalize_radians(angle: float) -> float:
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))


def is_inside_rect(
    rects: np.ndarray,
    width_of_outer_rect: int,
    height_of_outer_rect: int,
) -> np.ndarray:
    results = []
    for rect in rects:
        cx, cy, width, height, angle = rect
        if (cx < 0) or (cx > width_of_outer_rect):
            results.append(False)
        elif (cy < 0) or (cy > height_of_outer_rect):
            results.append(False)
        else:
            rect_tuple = ((cx, cy), (width, height), angle)
            box = cv2.boxPoints(rect_tuple)
            x_max = int(np.max(box[:, 0]))
            x_min = int(np.min(box[:, 0]))
            y_max = int(np.max(box[:, 1]))
            y_min = int(np.min(box[:, 1]))
            results.append(
                (x_min >= 0) and (x_max <= width_of_outer_rect)
                and (y_min >= 0) and (y_max <= height_of_outer_rect)
            )
    return np.asarray(results, dtype=np.bool_)


def bounding_box_from_rotated_rect(rects: np.ndarray) -> np.ndarray:
    results = []
    for rect in rects:
        cx, cy, width, height, angle = rect
        rect_tuple = ((cx, cy), (width, height), angle)
        box = cv2.boxPoints(rect_tuple)
        x_max = int(np.max(box[:, 0]))
        x_min = int(np.min(box[:, 0]))
        y_max = int(np.max(box[:, 1]))
        y_min = int(np.min(box[:, 1]))
        results.append([
            int((x_min + x_max) // 2),
            int((y_min + y_max) // 2),
            int(x_max - x_min),
            int(y_max - y_min),
            0,
        ])
    return np.asarray(results, dtype=np.float32)


def image_rotation_without_crop(
    images: List[np.ndarray],
    angles: np.ndarray,
) -> List[np.ndarray]:
    rotated_images = []
    for image, angle in zip(images, angles):
        height, width = image.shape[:2]
        image_center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, int(angle), 1)
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        rotation_matrix[0, 2] += bound_w / 2 - image_center[0]
        rotation_matrix[1, 2] += bound_h / 2 - image_center[1]
        rotated_images.append(
            cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h))
        )
    return rotated_images


def crop_rectangle(image: np.ndarray, rects: np.ndarray) -> List[np.ndarray]:
    cropped_images = []
    height, width = image.shape[:2]
    inside = is_inside_rect(rects, width, height)
    rects = rects[inside, ...]
    for rect in rects:
        cx, cy, rect_width, rect_height = (
            int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        )
        cropped_images.append(
            image[
                cy - rect_height // 2:cy + rect_height - rect_height // 2,
                cx - rect_width // 2:cx + rect_width - rect_width // 2,
            ]
        )
    return cropped_images


def rotate_and_crop_rectangle(
    image: np.ndarray,
    rects_tmp: np.ndarray,
    operation_when_cropping_out_of_range: str,
) -> List[np.ndarray]:
    rects = copy.deepcopy(rects_tmp)
    height, width = image.shape[:2]

    if operation_when_cropping_out_of_range == 'padding':
        size = (int(math.sqrt(width ** 2 + height ** 2)) + 2) * 2
        image = pad_image(image=image, resize_width=size, resize_height=size)
        rects[:, 0] = rects[:, 0] + abs(size - width) / 2
        rects[:, 1] = rects[:, 1] + abs(size - height) / 2
    elif operation_when_cropping_out_of_range == 'ignore':
        inside = is_inside_rect(rects, width, height)
        rects = rects[inside, ...]

    rect_bbx_upright = bounding_box_from_rotated_rect(rects=rects)
    rect_bbx_upright_images = crop_rectangle(image=image, rects=rect_bbx_upright)
    rotated_rect_bbx_upright_images = image_rotation_without_crop(
        images=rect_bbx_upright_images, angles=rects[..., 4:5],
    )

    rotated_cropped_images = []
    for rotated_image, rect in zip(rotated_rect_bbx_upright_images, rects):
        crop_cx = rotated_image.shape[1] // 2
        crop_cy = rotated_image.shape[0] // 2
        rect_width = int(rect[2])
        rect_height = int(rect[3])
        rotated_cropped_images.append(
            rotated_image[
                crop_cy - rect_height // 2:crop_cy + (rect_height - rect_height // 2),
                crop_cx - rect_width // 2:crop_cx + (rect_width - rect_width // 2),
            ]
        )
    return rotated_cropped_images


def keep_aspect_resize_and_pad(
    image: np.ndarray,
    resize_width: int,
    resize_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    image_height, image_width = image.shape[:2]
    padded_image = np.zeros((resize_height, resize_width, 3), np.uint8)
    ash = resize_height / image_height
    asw = resize_width / image_width
    if asw < ash:
        sizeas = (int(image_width * asw), int(image_height * asw))
    else:
        sizeas = (int(image_width * ash), int(image_height * ash))
    resized_image = cv2.resize(image, dsize=sizeas)
    start_h = int(resize_height / 2 - sizeas[1] / 2)
    end_h = int(resize_height / 2 + sizeas[1] / 2)
    start_w = int(resize_width / 2 - sizeas[0] / 2)
    end_w = int(resize_width / 2 + sizeas[0] / 2)
    padded_image[start_h:end_h, start_w:end_w, :] = resized_image.copy()
    return padded_image, resized_image


def pad_image(image: np.ndarray, resize_width: int, resize_height: int) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    resize_width = max(resize_width, image_width)
    resize_height = max(resize_height, image_height)
    padded_image = np.zeros((resize_height, resize_width, 3), np.uint8)
    start_h = int(resize_height / 2 - image_height / 2)
    end_h = int(resize_height / 2 + image_height / 2)
    start_w = int(resize_width / 2 - image_width / 2)
    end_w = int(resize_width / 2 + image_width / 2)
    padded_image[start_h:end_h, start_w:end_w, :] = image
    return padded_image
