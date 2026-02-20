from ultralytics import YOLO
import cv2
import numpy as np
import os
from src.pipeline.utils import *
from src.pipeline.plate_prep import *



_MODEL_FOOD = None

def get_food_model():
    global _MODEL_FOOD
    if _MODEL_FOOD is None:
        _MODEL_FOOD = YOLO('models/food_seg.pt')
    return _MODEL_FOOD


def volume_cal(capture_after_path,
               depth_after_path,
                capture_ctrl_path = 'demo\segment_and_volume\control\captures_color.jpeg',
                depth_ctrl_path = 'demo\segment_and_volume\control\captures_depth.png' ):

    img_control, depth_control,mask_control = plate_seg(capture_ctrl_path,depth_ctrl_path )
    img_after, depth_after, mask_after = plate_seg(capture_after_path, depth_after_path)

    centroid_control = get_centroid_from_mask(mask_control)
    centroid_after = get_centroid_from_mask(mask_after)


    img_after_shifted = shift_centroid(img_after,centroid_control,centroid_after)
    depth_after_shifted = shift_centroid(depth_after, centroid_control, centroid_after)
 

    
    model_food = get_food_model()
    results = model_food(img_after_shifted, conf = 0.5)

    volume, masks = vol_by_group(img_after_shifted, results, depth_control, depth_after_shifted)


    
    return volume



