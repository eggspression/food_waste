import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO
import sys
print(sys.argv)

def save_color_and_depth(folder='captures/captures_after', filename_prefix="captures"):
    
    
    os.makedirs(folder, exist_ok=True)

    hole_filling = rs.hole_filling_filter()
    dec_filter = rs.decimation_filter()
    spat_filter = rs.spatial_filter()
    temp_filter = rs.temporal_filter()

    # 2. Setup RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 3. Start camera
    pipeline.start(config)

    # Let camera warm up
    for _ in range(20):
        pipeline.wait_for_frames()

    try:
        # 4. Capture one aligned frame
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("Failed to get frames")
            return

        # 5. Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())  # raw depth in uint16


        # 6. Save color image
        color_path = os.path.join(folder, f"{filename_prefix}_color.jpeg")
        cv2.imwrite(color_path, color_image)
        print(f"Saved color image to {color_path}")
        
        # 7. Save depth image as visual PNG (for checking)
        depth_visual = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
        depth_img_path = os.path.join(folder, f"{filename_prefix}_depth_colored.jpeg")
        cv2.imwrite(depth_img_path, depth_colormap)
        print(f"Saved depth visualization to {depth_img_path}")

        # 8. Save depth image as raw .npy (for analysis)
        depth_npy_path = os.path.join(folder, f"{filename_prefix}_depth.npy")
        np.save(depth_npy_path, depth_image)
        print(f"Saved raw depth array to {depth_npy_path}")
        #9. Save depth image as png file
        depth_png_path = os.path.join(folder, f"{filename_prefix}_depth.png")
        cv2.imwrite(depth_png_path, depth_image)  # depth_image is uint16

    finally:
        pipeline.stop()

if __name__ == "__main__":
    save_color_and_depth()
