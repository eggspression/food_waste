from ultralytics import YOLO
import cv2
import numpy as np
import os
from src.pipeline.utils import resize_with_padding_img

def extract_contour_mainplate(results):
    """
    Finds the contour of the main plate of the tray 

    Parameters:
        mask (np.array): Binary image with object as white (255) on black (0) background.

    Returns:
        best_contour (np.array): array of the points of the contour around the main plate
    """
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    max_area = 0
    best_contour = None
    for i, c in enumerate(results[0].masks.xy):
            class_id = class_ids[i]
            if class_id == 0 or class_id == 1:
                contour = c.astype(np.int32)
                area = cv2.contourArea(contour)
                print(area)
                if area > max_area:
                    max_area = area
                    best_contour = contour
    return best_contour

def apply_mask_by_dtype(img, contour):
    """
     Seperate out the main plate with the contour given along with enhancing the contour to be more precise

    Parameters:
        img: the image treating
        contour (np.array): array of the coordinates of points of the contour

    Returns:
        masked_img: image of the main plate (points in the contour) other pixels are black
        clean_masl: the binary mask of the main plate
    """
     
    contour_int = contour.astype(np.int32)  # Ensure correct type        
    mask = np.zeros(img.shape[:2], dtype=np.uint8) 
    cv2.drawContours(mask, [contour_int], -1, 255, thickness=-1)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=-1)
    else:
        clean_mask = mask  
    if len(largest) >= 5:
        ellipse = cv2.fitEllipse(largest)
        ellipse_mask = np.zeros_like(mask)
        cv2.ellipse(ellipse_mask, ellipse, 255, thickness=-1)
        clean_mask = cv2.bitwise_and(clean_mask, ellipse_mask)

    if img.dtype == np.uint8 and img.ndim == 3:
        return cv2.bitwise_and(img, img, mask=clean_mask), clean_mask

    elif img.dtype in [np.uint16, np.float32] and img.ndim == 2:
        background_val = 0 if img.dtype == np.uint16 else np.nan
        return np.where(clean_mask == 255, img, background_val), clean_mask

    else:
        raise ValueError(f"Unsupported dtype: {img.dtype} or shape: {img.shape}")
    
_MODEL_PLATE = None

def get_plate_model():
    global _MODEL_PLATE
    if _MODEL_PLATE is None:
        _MODEL_PLATE = YOLO('models/plate_seg.pt')
    return _MODEL_PLATE

def plate_seg(rgb_path, depth_path):

    img = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)


    img_resized,geo = resize_with_padding_img(img)
    model_plate = get_plate_model()
    results = model_plate(img_resized, conf = 0.5)
    contour_resized = extract_contour_mainplate(results=results)
    masked_img_resized, clean_mask_resized = apply_mask_by_dtype(img = img_resized, contour=contour_resized)
    mask_crop = clean_mask_resized[geo['top']:geo['top']+geo['new_h'],
                     geo['left']:geo['left']+geo['new_w']]
    clean_mask_original = cv2.resize(mask_crop,
                           (geo['orig_w'], geo['orig_h']),
                           interpolation=cv2.INTER_NEAREST)
    masked_depth = np.where(clean_mask_original == 255, depth, 0)
    masked_img_original = cv2.bitwise_and(img,img, mask=clean_mask_original)
    

    return masked_img_original, masked_depth, clean_mask_original

def plate_seg_no_depth(rgb_path):

    img = cv2.imread(rgb_path)

    img_resized,geo = resize_with_padding_img(img)
    model_plate = YOLO('models/plate_seg.pt')
    results = model_plate(img_resized, conf = 0.5)
    contour_resized = extract_contour_mainplate(results=results)
    masked_img_resized, clean_mask_resized = apply_mask_by_dtype(img = img_resized, contour=contour_resized)
    mask_crop = clean_mask_resized[geo['top']:geo['top']+geo['new_h'],
                     geo['left']:geo['left']+geo['new_w']]
    clean_mask_original = cv2.resize(mask_crop,
                           (geo['orig_w'], geo['orig_h']),
                           interpolation=cv2.INTER_NEAREST)
    masked_img_original = cv2.bitwise_and(img,img, mask=clean_mask_original)
    

    return masked_img_original, clean_mask_original






if __name__ == "__main__":
    plate_seg()