import numpy as np
import cv2

def generate_binary_mask(mask):
    """
    Binarizes a given mask that can be in RGB format or not normalized.
    
    Parameters
    ----------
    mask : mask to binarize
    
    Returns
    -------
    mask : binarized and normalized mask (0 and 1)
    """
    
    # if mask is RGB => to grayscale
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # if mask is not binary, we transform it
    if not(np.unique(mask) == (0, 1)).all():
        mask = (mask == 255).astype(float)
    return mask