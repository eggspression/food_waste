# Food Waste Estimation using RGB + Depth (AI Pipeline)

## Overview

This project implements an end-to-end computer vision pipeline to estimate food waste volume from tray images.
Using:
- RGB images
- Depth images (Intel RealSense D415)
- Deep learning segmentation (YOLOv11-seg)

The system:
Segments the main plate
Segments food by category
Aligns depth captures with a calibrated reference
Computes per-class and total food volume
The objective is to demonstrate how computer vision and depth sensing can be combined for automated tray analysis in collective catering environments.

## Processing Pipeline
```
RGB Image + Depth Image
        │
        ▼
[1] Plate Segmentation (YOLOv11-seg)
        │
        ▼
Plate-only RGB Image (background removed)
        │
        ├───────────────────────────────────┐
        │                                   │
        ▼                                   ▼
[2] Food Segmentation(YOLOv11-seg)   [3] Plate Alignment (Depth)
        │                                   │
Food Masks by Class                         Aligned Depth Maps
        │                                   │
        └───────────────────────────────┐───┴
                                        ▼
                          [4] Volume Computation
                                        ▼
                        Volume per Class + Total Volume

```

## Step-by-Step Explanation

### 1. Plate Segmentation

Input: Full RGB tray image  
Output: Plate-only RGB image + plate mask  

A YOLOv11 segmentation model detects the main circular plate.  
A binary plate mask is generated and applied to the RGB image.  
All pixels outside the plate are set to black.

Result:

```
RGB → Plate-only RGB
```

This removes tray background, cutlery, glasses, and surrounding noise.

---

### 2. Food Segmentation

Input: Plate-only RGB image  
Output: Binary mask per food class  

A second YOLOv11-seg model segments food regions inside the plate.

Classes:
- Féculents (starches)
- Viande (meat)
- Légumes (vegetables)
- Déchets (waste)

The model outputs segmentation overlays and per-class binary masks, which are later used for volume estimation.

---

### 3. Plate Alignment (Depth Calibration)

Volume estimation requires alignment between:

- Current depth image (meal tray)
- Reference depth image (empty plate)

Alignment process:

1. Extract plate mask from both captures.
2. Compute centroid of each plate mask.
3. Translate one depth matrix to align centroids.
4. Ensure pixel-wise correspondence.

This compensates for small spatial shifts during acquisition.

---

### 4. Volume Estimation

After depth alignment:

#### Depth Subtraction

For each pixel inside the plate mask:

```
Δdepth = ReferenceDepth - CurrentDepth
```

Only positive differences are considered (food above plate surface).

#### Pixel-to-Volume Conversion

Each pixel represents a small 3D column.

Volume is estimated as:

```
Volume_pixel = Δdepth × (pixel_size)^2
```

Where:
- Δdepth = depth difference (mm)
- pixel_size = real-world size of one pixel (derived from camera geometry)

#### Class-wise Volume

Using food segmentation masks:

```
Volume_class = Σ Volume_pixel over pixels in that class
Total_volume = Σ Volume_pixel over entire plate mask
```

Final output:
- Volume per food category
- Total leftover volume

---

## Model Performance

### Plate Segmentation
- ~100 annotated images
- mAP50 ≈ ~0.90

### Food Segmentation
- 400 training images
- 50 validation images
- mAP50 ≈ ~0.58
- mAP50-95 ≈ ~0.36

Performance is limited by small dataset size, overlapping food, and visual similarity between categories.

---

## Project Structure

```
run_app.py              # Entry point
src/
  ui/                   # Tkinter GUI
  pipeline/             # Segmentation + alignment + volume logic
models/                 # Trained YOLO weights
demo/                   # Example images
requirements.txt
```

---

## Installation

```bash
git clone https://github.com/eggspression/food_waste
cd food-waste
pip install -r requirements.txt
```

---

## Running the Application

```bash
py run_app.py
```

The interface allows:
- Segmentation-only mode (RGB only)
- Segmentation + volume mode (RGB + depth + reference depth)

Demo files are available inside the `demo/` directory.

---

## Assumptions and Limitations

- Camera height must remain constant (~70 cm during experiments).
- Reference depth capture must match acquisition setup.
- Plate must be approximately centered.
- Overlapping food reduces segmentation accuracy.
- Depth resolution introduces small geometric approximation errors.
- The system is calibrated for a specific RealSense D415 setup.

This prototype demonstrates feasibility rather than production-level robustness.

---


## Academic Context

Developed as an engineering project at INSA Lyon (2024–2025), during the second semester of the second year for the P2I Project.