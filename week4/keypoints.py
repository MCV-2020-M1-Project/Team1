import cv2
from enum import Enum
import numpy as np
from skimage import feature
from typing import List
import os

def _keypoint_detector(image:np.ndarray, mode=None):
    """
    Extract keypoints from an image

    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mode:  indicates if keypoints are detected on a query or a database image

    Returns:
        (list of cv2.KeyPoint objects): list of keypoints.
    """
    pass

class Mode(Enum):
    QUERY = 0
    IMAGE = 1

def laplacian_of_gaussian(image):
    """
    Extract keypoints of an image using Laplacian of Gaussians method.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
    Returns:
        (list of cv2.KeyPoint objects): list of keypoints.
    """

    blobs_log = feature.blob_log(image, max_sigma=30, num_sigma=10, threshold=.1)
    points2f = np.array(blobs_log[:, [0, 1]], dtype=np.float32)
    sizes = np.array(list(blobs_log[:, 2] * np.sqrt(2) * 2), dtype=np.float32)
    keypoints = []
    for i in range(len(points2f)):
        keypoints.append(cv2.KeyPoint_convert([points2f[i]], sizes[i])[0])
    return keypoints

def difference_of_gaussian(image):
    """
    Extract keypoints of an image using Difference of Gaussians method.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
    Returns:
        (list of cv2.KeyPoint objects): list of keypoints.
    """

    blobs_dog = feature.blob_dog(image, max_sigma=30, threshold=.1)
    points2f = np.array(blobs_dog[:, [0, 1]], dtype=np.float32)
    sizes = np.array(list(blobs_dog[:, 2] * np.sqrt(2) * 2), dtype=np.float32)
    keypoints = []
    for i in range(len(points2f)):
        keypoints.append(cv2.KeyPoint_convert([points2f[i]], sizes[i])[0])
    return keypoints

def determinant_of_hessian(image):
    """
    Extract keypoints of an image using Determinant of Hessian method.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
    Returns:
        (list of cv2.KeyPoint objects): list of keypoints.
    """

    blobs_doh = feature.blob_doh(image, max_sigma=30, threshold=.001)
    points2f = np.array(blobs_doh[:, [0, 1]], dtype=np.float32)
    sizes = np.array(list(blobs_doh[:, 2] * np.sqrt(2) * 2), dtype=np.float32)
    keypoints = []
    for i in range(len(points2f)):
        keypoints.append(cv2.KeyPoint_convert([points2f[i]], sizes[i])[0])
    return keypoints

def harris_laplacian(image):
    """
    Extract keypoints of an image using the Harris-Laplace feature detector as described
    in "Scale & Affine Invariant Interest Point Detectors" (Mikolajczyk and Schimd).
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
    Returns:
        (list of cv2.KeyPoint objects): list of keypoints.
    """

    hl = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
    keypoints = hl.detect(image)
    return keypoints

def sift_keypoints(image, mode=0):
    """
    Extract keypoints of an image using Difference of Gaussians method.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mode (int): indicates if keypoints are detected on a query or a database image.
    Returns:
        (list of cv2.KeyPoint objects): list of keypoints.
    """

    if mode == Mode.QUERY:
        nkeypoints = 200
    elif mode == Mode.IMAGE:
        nkeypoints = 100
    else:
        nkeypoints = 0

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=nkeypoints)
    keypoints = sift.detect(image)
    return keypoints

def surf_keypoints(image, mode=0):
    """
    Extract keypoints of an image using Box Filter to approximate LoG, and the
    Hessian matrix for both scale and location.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mode (int): indicates if keypoints are detected on a query or a database image.
    Returns:
        (list of cv2.KeyPoint objects): list of keypoints.
    """

    if mode == Mode.QUERY:
        hessian_thresh = 400
    elif mode == Mode.IMAGE:
        hessian_thresh = 1000
    else:
        hessian_thresh = 400

    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_thresh)
    keypoints = surf.detect(image)
    return keypoints

def orb_keypoints(image):
    """
    Extract keypoints of an image using the ORB method.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
    Returns:
        list (list of object keypoint): list of keypoints.
    """

    orb = cv2.ORB_create()
    keypoints = orb.detect(image)
    return keypoints

def harris_corner_detector(image, mode=0):

    """
    Extract keypoints from image using Harris Corner Detector.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
    Returns:
        ndarray: list of 1D arrays of type np.float32 containing image descriptors.
    """

    dst = cv2.cornerHarris(image, 4, -1, 0.04)
    corners = np.argwhere(dst > dst.max() * 0.10)
    return [cv2.KeyPoint(corner[0], corner[1], 9) for corner in corners]

def harris_corner_subpixel_accuracy(image, mode=0):
    """
    Extract keypoints from image using Harris Corner Detector with subpixel accuracy.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
    Returns:
        ndarray: list of 1D arrays of type np.float32 containing image descriptors.
    """

    if mode == Mode.QUERY:
        thresh = 0.15
    elif mode == Mode.IMAGE:
        thresh = 0.10
    else:
        thresh = 0.15

    # find Harris corners
    dst = cv2.cornerHarris(image, 4, -1, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, thresh * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(image, np.float32(centroids), (2, 2), (-1, -1), criteria)

    return [cv2.KeyPoint(corner[0], corner[1], 4) for corner in corners]

KEYPOINT_METHODS = {
    'sift': sift_keypoints,
    'surf': surf_keypoints,
    'orb': orb_keypoints,
    'harris_corner_detector': harris_corner_detector,
    'harris_corner_subpixel': harris_corner_subpixel_accuracy,
    'hl': harris_laplacian,
    'dog': difference_of_gaussian,
    'log': laplacian_of_gaussian,
    'doh': determinant_of_hessian
    }

def extract_keypoints(image:np.ndarray, method:str, mode:int=None):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if mode is not None:
        return KEYPOINT_METHODS[method](image, mode)
    else:
        return KEYPOINT_METHODS[method](image)

if __name__ == '__main__':
    image = cv2.imread(os.path.join('images','barca.png'))
    print(f'image shape: {image.shape}')
    print('KEYPOINT METHODS')
    for key in KEYPOINT_METHODS:
        print(f'keypoint method: {key}, num_keypoints = {len(extract_keypoints(image, method=key))}') 