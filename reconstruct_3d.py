# src/reconstruct_3d.py
# replace or update in src/reconstruct_3d.py (import numpy etc. already vorhanden)
def volume_to_mesh(volume, spacing=(1.0,1.0,1.0), out_path="results/lung_mesh.stl", level=0.5):
    # volume: (Z,H,W) binary mask or intensity
    import numpy as np
    from skimage import measure
    import pyvista as pv

    # ensure binary mask
    if volume.dtype != np.bool_:
        vol_bin = (volume > level).astype(np.uint8)
    else:
        vol_bin = volume.astype(np.uint8)

    # remove slices with almost no mask at ends (trim empty slices)
    slice_sums = vol_bin.reshape(vol_bin.shape[0], -1).sum(axis=1)
    # threshold: keep slices with >0 pixels (or adjust)
    keep = np.where(slice_sums > 0)[0]
    if keep.size == 0:
        raise RuntimeError("Empty mask: nothing to reconstruct")
    z0, z1 = int(keep[0]), int(keep[-1])
    vol_trim = vol_bin[z0:z1+1]

    # compute 3D bbox of foreground inside trimmed volume
    zs, ys, xs = np.where(vol_trim)
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    # crop to bbox (y,x)
    vol_crop = vol_trim[:, ymin:ymax+1, xmin:xmax+1]

    # also compute new spacing accordingly; spacing = (dz, dy, dx)
    dz, dy, dx = spacing
    # adjust spacing remains same but grid size changes

    # optional: morphological closing / small object removal BEFORE MC (if noisy)
    try:
        from skimage.morphology import closing, ball
        vol_crop = closing(vol_crop.astype(bool), ball(1)).astype(np.uint8)
    except Exception:
        pass

    # marching cubes on cropped volume
    verts, faces, normals, values = measure.marching_cubes(vol_crop, level=0.5, spacing=(dz, dy, dx))
    # translate verts to original coordinates: add offsets
    # x offset = xmin * dx, y offset = ymin * dy, z offset = z0 * dz
    verts[:, 0] += z0 * dz   # careful: skimage returns verts in (z, y, x) order matching spacing
    verts[:, 1] += ymin * dy
    verts[:, 2] += xmin * dx

    # convert faces for pyvista
    faces_pv = np.hstack([np.full((faces.shape[0],1), 3), faces]).astype(np.int64)
    faces_flat = faces_pv.reshape(-1)
    mesh = pv.PolyData(verts, faces_flat)

    # optional smoothing & decimation
    try:
        mesh = mesh.smooth(n_iter=20)  # PyVista smoothing (if available)
    except Exception:
        pass

    mesh.save(out_path)
    print("Mesh gespeichert:", out_path)
    return mesh
