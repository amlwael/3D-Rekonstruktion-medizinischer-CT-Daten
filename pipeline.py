# src/pipeline.py
import os
import numpy as np
import cv2
from load_dicom import load_dicom_series
from segment_lung import segment_lungs
from reconstruct_3d import volume_to_mesh
import matplotlib.pyplot as plt

def ensure_dirs():
    os.makedirs("results", exist_ok=True)

def run_pipeline(dicom_folder="data/anonym/patient11"):
    ensure_dirs()
    print("Load DICOMs from:", dicom_folder)
    vol, spacing = load_dicom_series(dicom_folder)  # vol: (Z,H,W)
    print("Volume shape (Z,H,W):", vol.shape, "spacing (dz,dy,dx):", spacing)
    # For simple segmentation we operate slice-wise or volume-wise depending on your function
    print("Segmentation ...")
    mask_vol = segment_lungs(vol)  # expects volume shape (Z,H,W)
    # save an overlay of middle slice
    mid = vol.shape[0] // 2
    img = vol[mid]
    mask = (mask_vol[mid] > 0).astype(np.uint8) * 255
    overlay = cv2.cvtColor(((img - img.min())/(img.max()-img.min())*255).astype('uint8'), cv2.COLOR_GRAY2BGR)
    # draw contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0,0,255), 1)
    cv2.imwrite("results/segmentation_overlay.png", overlay)
    print("Saved overlay: results/segmentation_overlay.png")
    # reconstruct mesh using spacing (dz,dy,dx)
    mesh = volume_to_mesh(mask_vol, spacing=spacing, out_path="results/lung_mesh.stl")
    # Try to create a screenshot if pyvista GUI not available
    try:
        p = mesh.plot(off_screen=True, screenshot="results/mesh_snapshot.png")
        print("Screenshot geschrieben: results/mesh_snapshot.png")
    except Exception as e:
        print("Info: konnte Screenshot nicht erstellen:", e)
    print("Pipeline fertig.")

if __name__ == "__main__":
    run_pipeline()
