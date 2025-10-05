import os
import pydicom
import numpy as np

def _get_sort_key(ds):
    # bevorzugt ImagePositionPatient (z-Koordinate), sonst InstanceNumber
    if hasattr(ds, "ImagePositionPatient"):
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            pass
    if hasattr(ds, "InstanceNumber"):
        try:
            return int(ds.InstanceNumber)
        except Exception:
            pass
    # fallback: SOPInstanceUID
    return ds.SOPInstanceUID if hasattr(ds, "SOPInstanceUID") else 0

def _apply_rescale(ds, arr):
    # CT DICOMs brauchen oft RescaleSlope/Intercept -> Hounsfield Units
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr.astype(np.float32) * slope + intercept

def load_dicom_series(folder):
    """
    Lädt eine Serie von DICOMs aus 'folder' und gibt ein 3D-Volume sowie Spacing zurück.
    Vereinheitlicht Bildgrößen per Zero-Padding, sortiert nach Slice-Position.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = [f for f in os.listdir(folder) if f.lower().endswith(".dcm")]
    if len(files) == 0:
        raise FileNotFoundError(f"No DICOM files in folder: {folder}")

    # read datasets
    dsets = []
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            ds = pydicom.dcmread(path, force=True)
            dsets.append((path, ds))
        except Exception as e:
            print(f"Warning: konnte DICOM nicht lesen: {path} -> {e}")

    if len(dsets) == 0:
        raise RuntimeError("Keine lesbaren DICOMs gefunden.")

    # sort datasets by z-position or InstanceNumber
    dsets.sort(key=lambda x: _get_sort_key(x[1]))

    # collect pixel arrays and track sizes
    imgs = []
    shapes = []
    spacings = []
    for path, ds in dsets:
        arr = ds.pixel_array
        if arr.ndim == 3:
            arr = arr[..., 0]  # nur 2D behalten
        arr = _apply_rescale(ds, arr)
        imgs.append((path, ds, arr))
        shapes.append(arr.shape)
        ps = getattr(ds, "PixelSpacing", None)
        st = getattr(ds, "SliceThickness", None)
        if ps is not None and st is not None:
            try:
                spacing = (float(st), float(ps[0]), float(ps[1]))  # (z,y,x)
            except:
                spacing = None
        else:
            spacing = None
        spacings.append(spacing)

    # Zielgröße bestimmen
    heights = [s[0] for s in shapes]
    widths  = [s[1] for s in shapes]
    H = max(heights)
    W = max(widths)

    # Volumen anlegen
    Z = len(imgs)
    vol = np.zeros((Z, H, W), dtype=np.float32)

    for i, (path, ds, arr) in enumerate(imgs):
        h, w = arr.shape
        top = (H - h) // 2
        left = (W - w) // 2
        vol[i, top:top+h, left:left+w] = arr

    # mittleres spacing wählen
    valid_spacings = [s for s in spacings if s is not None]
    if len(valid_spacings) > 0:
        z_vals = [s[0] for s in valid_spacings]
        y_vals = [s[1] for s in valid_spacings]
        x_vals = [s[2] for s in valid_spacings]
        spacing = (float(np.median(z_vals)),
                   float(np.median(y_vals)),
                   float(np.median(x_vals)))
    else:
        spacing = (1.0, 1.0, 1.0)

    return vol, spacing
