import numpy as np
from skimage import measure, morphology

def segment_lungs(volume, threshold=-320):
    # Threshold: Luft = dunkel, Gewebe = heller
    binary = volume > threshold
    # Remove small noise
    cleaned = morphology.remove_small_objects(binary, min_size=500)
    return cleaned.astype(np.uint8)
