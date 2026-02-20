import numpy as np
from src.pipeline.utils import *
import matplotlib.pyplot as plt
from src.pipeline.plate_prep import plate_seg



def show_total_volume(capture_after_path,
               depth_after_path,
                capture_ctrl_path = 'demo\segment_and_volume\control\captures_color.jpeg',
                depth_ctrl_path = 'demo\segment_and_volume\control\captures_depth.png' ):
    

    img_control, depth_control,mask_control = plate_seg(capture_ctrl_path,depth_ctrl_path )
    img_after, depth_after, mask_after = plate_seg(capture_after_path, depth_after_path)

    centroid_control = get_centroid_from_mask(mask_control)
    centroid_after = get_centroid_from_mask(mask_after)

    img_after_shifted = shift_centroid(img_after,centroid_control,centroid_after)
    depth_after_shifted = shift_centroid(depth_after, centroid_control, centroid_after)

    obj_loc = np.full(depth_control.shape, 0)
    vol = 0.0
    zonex = range(min(depth_control.shape[0], depth_after_shifted.shape[0]))
    zoney = range(min(depth_control.shape[1], depth_after_shifted.shape[1]))



    dc  = depth_control.astype(np.float64)
    das = depth_after_shifted.astype(np.float64)
    for i in zonex:
        for j in zoney:
            d = dc[i,j]-das[i,j]
            if d>0 and d < 100: 
                obj_loc[i,j] = d
                vol += d*depth_to_pixel_size((das[i,j]+dc[i,j])/2)**2
                
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)

    im = ax.imshow(obj_loc, vmin=0, vmax=50, cmap="viridis")
    ax.axis("off")                                    

    cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                        fraction=0.046, pad=0.04)     
    cbar.set_label("Δ profondeur (mm)")

    fig.tight_layout()

    out = fig_to_image(fig, width_px=1280, height_px=720)
    plt.close(fig)  

    return out