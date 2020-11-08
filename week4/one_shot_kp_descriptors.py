import cv2
import numpy as np


def sift_descriptors(image, mask):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (256, 256), interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)

    return cv2.xfeatures2d.SIFT_create().detectAndCompute(gray_img, mask=mask)[1]


def surf_descriptors(image, mask):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (256, 256), interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)

    return cv2.xfeatures2d.SURF_create().detectAndCompute(gray_img, mask)[1]


def root_sift_descriptors(image, mask, eps=1e-7):
    descs = sift_descriptors(image, mask)
    if descs is not None:
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
    
    return descs


def orb_descriptors(image, mask):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_img = cv2.resize(gray_img, (256, 256), interpolation=cv2.INTER_AREA)
    #if mask is not None:
    #    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)

    return cv2.ORB_create(nfeatures=1200, WTA_K=2).detectAndCompute(gray_img, mask)[1]


def daisy_descriptors(image, mask):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (256, 256), interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)

    detector = cv2.FastFeatureDetector_create()
    keypoints = detector.detect(gray_img ,mask)
    
    return cv2.xfeatures2d.DAISY_create().compute(gray_img, keypoints)[1]


def brisk_descriptors(image, mask):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (256, 256), interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)

    return cv2.BRISK_create().detectAndCompute(gray_img, mask)[1]


LOCAL_DESCRIPTORS = {
    'sift': sift_descriptors,
    'surf': surf_descriptors,
    'root_sift': root_sift_descriptors,
    'orb': orb_descriptors,
    'daisy': daisy_descriptors,
    'brisk': brisk_descriptors,
}

def extract_local_descriptors(image, mask, method='sift'):
    return LOCAL_DESCRIPTORS[method](image, mask)