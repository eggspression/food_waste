import cv2
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt



def resize_with_padding_img(image, desired_size=960):
    old_h, old_w = image.shape[:2]
    ratio = float(desired_size) / max(old_h, old_w)
    new_w, new_h = int(old_w * ratio), int(old_h * ratio)

    resized = cv2.resize(image, (new_w, new_h))
    delta_w, delta_h = desired_size - new_w, desired_size - new_h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=[114,114,114])

    geo = dict(ratio=ratio, top=top, left=left,
               new_h=new_h, new_w=new_w,
               orig_h=old_h, orig_w=old_w)
    return padded, geo


def resize_with_padding_depth(depth, desired_size = 960):
    old_size = depth.shape[:2]  
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    resized_image = cv2.resize(depth, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [114, 114, 114]
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=color)

    return new_image



def depth_to_pixel_size(d):
    return 0.0011 * d - 0.0124



def fig_to_image(fig=None, *, width_px=1280, height_px=720, dpi=100):
    """
    Convert a Matplotlib figure to an RGB NumPy array with a precise resolution.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or None
        The figure to convert.  If None, uses `plt.gcf()` (current figure).
    width_px : int
        Desired output width in pixels.
    height_px : int
        Desired output height in pixels.
    dpi : int
        Dots-per-inch used when rendering.  The actual pixel size is forced by
        width_px / dpi and height_px / dpi.

    Returns
    -------
    np.ndarray
        3-channel uint8 array, shape (height_px, width_px, 3).
    """
    if fig is None:
        fig = plt.gcf()

    fig.set_size_inches(width_px / dpi, height_px / dpi)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    return np.array(img)



def get_centroid_from_mask(mask):
    """
    Finds the centroid (center) of the largest object in a binary mask.

    Parameters:
        mask (np.array): Binary image with object as white (255) on black (0) background.

    Returns:
        tuple: (cx, cy) coordinates of the centroid.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the mask.")
    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        raise ValueError("no centroid TT")
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy)

def shift_centroid(after_plate, ctrl_centroid, after_centroid):
    dx = ctrl_centroid[0] - after_centroid[0]
    dy = ctrl_centroid[1] - after_centroid[1]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(after_plate, M ,(after_plate.shape[1],after_plate.shape[0]),borderMode=cv2.BORDER_CONSTANT,flags = cv2.INTER_NEAREST, borderValue=0)
    return shifted

def vol_by_group(img, results, depth_control, depth_after):
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    volume = [0.0] * (max(class_ids)+1)
    masks = []
    #convert to float from uint16 for calculations
    dc  = depth_control.astype(np.float64) 
    das = depth_after.astype(np.float64)
    
    for i,c in enumerate(results[0].masks.xy):
        contour = c.astype(np.int32)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        masks.append(mask)
        
        diff = dc - das
        valid = (diff > 0) & (diff < 50) & (mask>0) 
        pixel_size  = depth_to_pixel_size(das)
        vol = np.sum(diff[valid]*pixel_size[valid]**2)
        volume[class_ids[i]] += vol
    return volume, masks