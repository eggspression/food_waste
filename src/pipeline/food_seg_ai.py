from ultralytics import YOLO
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time

def masks_by_group(img, results):
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    masks = []

    for i,c in enumerate(results[0].masks.xy):
        contour = c.astype(np.int32)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        masks.append(mask)
    return masks

_MODEL_FOOD = None

def get_food_model():
    global _MODEL_FOOD
    if _MODEL_FOOD is None:
        _MODEL_FOOD = YOLO('models/food_seg.pt')
    return _MODEL_FOOD

def food_seg(img_after):
    model_food = model_food = get_food_model()
    img_after = cv2.resize(img_after, (960,960))
    results = model_food(img_after, conf = 0.4)
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = model_food.names
    masks = masks_by_group(img_after, results)
    fig = results[0].plot()
    return fig




