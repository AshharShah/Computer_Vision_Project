# 3D Scene Reconstruction & Virtual Tour

### CS436 --- Computer Vision Fundamentals · Final Project

This project implements a modular **Incremental Structure-from-Motion
(SfM)** pipeline for reconstructing a 3D scene from a sequence of
images. The system recovers both **camera poses** and **3D point
geometry** by starting with a two-view reconstruction and incrementally
adding new views using **PnP**, while periodically refining the entire
model with **Global Bundle Adjustment (GBA)**.

## Features

- **Keypoint detection & feature matching** (FLANN)
- **Essential matrix estimation** & initial pose recovery
- **Incremental camera registration** using PnP
- **3D point triangulation** for every new camera
- **Global bundle adjustment** (scene-wide optimization)
- **Export utilities** for:
  - `cameras.json`
  - `points.json`
  - `view_graph.json`
- **Interactive 3D visualization** of point clouds (Plotly)
- Modularized, clean implementation using multiple `.py` files

## Dependencies

This project requires:

- Python 3.x\
- OpenCV (`opencv-python`)\
- NumPy\
- Matplotlib\
- Plotly\
- SciPy

Install all dependencies with:

```bash
pip install numpy opencv-python matplotlib plotly scipy
```

## Input Images (Pre-Processed)

All input images for the SfM pipeline are stored here:

    Tehzeeb_v2/resized_images/

These images were prepared during **Week 1** of development in the
`Submission_Notebooks` folder.

The pipeline loads images from this path (relative to `Main_Pipeline/`):

```python
output_dir = r"../Tehzeeb_v2/resized_images"
```

## Repository Structure

    root/
    │
    ├── Submission_Notebooks/
    │   ├── Week1.ipynb
    │   ├── Week2.ipynb
    │   ├── Week3.ipynb
    │   └── (Development notebooks with explanations & visualizations)
    │
    └── Main_Pipeline/
        ├── sfm_export.py     # JSON export utilities
        ├── sfm_utils.py      # Matching, visualizations, helper functions
        ├── sfm_solver.py     # Geometry: triangulation, PnP, bundle adjustment
        ├── main.py           # Full pipeline (script version)
        └── main.ipynb        # Jupyter notebook version with visualization

## Running the Pipeline

### Option 1 --- Python Script

```bash
cd Main_Pipeline
python3 main.py
```

### Option 2 --- Jupyter Notebook

```bash
jupyter notebook main.ipynb
```

## Output Files

File Description

---

`cameras.json` All recovered camera intrinsics & poses
`points.json` Reconstructed 3D point cloud
`view_graph.json` View graph showing camera connections
