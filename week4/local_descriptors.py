from __future__ import division

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def sift_descriptors(image, keypoints):
    """
    Extract descriptors from keypoints using the SIFT method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        keypoints (list): list of cv2.KeyPoint objects.

    Returns:
        descriptors (ndarray): 2D array of type np.float32 and shape (#keypoints x 128)
            containing local descriptors for the keypoints.

    """

    sift = cv2.xfeatures2d.SIFT_create()
    _, descriptors = sift.compute(image, keypoints)
    return descriptors

def surf_descriptors(image, keypoints):
    """
    Extract descriptors from keypoints using the SURF method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        keypoints (list): list of cv2.KeyPoint objects.

    Returns:
        descriptors (ndarray): 2D array of type np.float32 and shape (#keypoints x 64)
            containing local descriptors for the keypoints.

    """

    surf = cv2.xfeatures2d.SURF_create()
    _, descriptors = surf.compute(image, keypoints)
    return descriptors

def root_sift_descriptors(image, keypoints, eps=1e-7):
    """
    Extract descriptors from keypoints using the RootSIFT method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        keypoints (list): list of cv2.KeyPoint objects.

    Returns:
        descriptors (ndarray): 2D array of type np.float32 containing local descriptors for the keypoints.

    """

    descs = sift_descriptors(image, keypoints)
    descs /= (descs.sum(axis=1, keepdims=True) + eps)
    descs = np.sqrt(descs)

    return descs

def orb_descriptors(image, keypoints):
    """
    Extract descriptors from keypoints using the ORB method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        keypoints (list): list of cv2.KeyPoint objects.

    Returns:
        descriptors (ndarray): 2D array of type np.float32 containing local descriptors for the keypoints.

    """

    orb = cv2.ORB_create(WTA_K=4)
    _, descriptors = orb.compute(image, keypoints)
    return descriptors

def daisy_descriptors(image, keypoints):
    """
    Extract descriptors from keypoints using the Daisy method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        keypoints (list): Not used

    Returns:
        descriptors (ndarray): 2D array of type np.float32 containing local descriptors for the keypoints.

    """

    daisy = cv2.xfeatures2d.DAISY_create()
    _, descriptors = daisy.compute(image, keypoints)
    return descriptors

def lbp(image, keypoints):
    """
    Extract descriptors from keypoints using the Local Binary Pattern method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        keypoints (list): Not used

    Returns:
        descriptors (ndarray): 2D array of type np.float32 containing local descriptors for the keypoints.

    """

    result = []
    for kp in keypoints:
        img = image[round(kp.pt[1] - kp.size/2):round(kp.pt[1] + kp.size/2),
        round(kp.pt[0] - kp.size/2):round(kp.pt[0] + kp.size/2)]

        numPoints = 30
        radius = 2
        eps = 1e-7

        lbp = local_binary_pattern(img, numPoints, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        result.append(np.array(hist, dtype=np.float32))

    # return the histogram of Local Binary Patterns
    return result

def hog_descriptor(image, keypoints):
    """
    Extract descriptors from keypoints using the Histogram of Gradients method.

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        keypoints (list): Not used

    Returns:
        descriptors (ndarray): 2D array of type np.float32 containing local descriptors for the keypoints.

    """
    hog = cv2.HOGDescriptor()
    result = []
    for kp in keypoints:
        descriptor = hog.compute(image, locations=[kp.pt])
        if descriptor is None:
            descriptor = []
        else:
            descriptor = descriptor.ravel()
        result.append(np.array(descriptor, dtype=np.float32))
    return result

LOCAL_DESCRIPTORS = {
    'sift': sift_descriptors,
    'surf': surf_descriptors,
    'root_sift': root_sift_descriptors,
    'orb': orb_descriptors,
    'daisy': daisy_descriptors,
    'hog': hog_descriptor,
    'lbp': lbp
}

def extract_local_descriptors(image, keypoints, method):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return LOCAL_DESCRIPTORS[method](image, keypoints)
