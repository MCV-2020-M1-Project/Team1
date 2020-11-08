import numpy as np
import cv2

def lowe_filter(matches, k=0.6):
    filtered = []
    for m, n in matches:
        if m.distance < k * n.distance:
            filtered.append(m)
    return filtered


def keypoints_based_similarity(matches, min_matches=5, max_dist=800):
    m = len(matches)
    # we compute the mean of distances among all keypoint matches
    d = np.mean([match.distance for match in matches]) if m > 0 else np.inf
    # we check minimum of matches and maximum distance thresholds
    if m < min_matches or d > max_dist:
        return 0.0
    return m / d # metric: matches per unit of distance


def bruteforce_matching(u, v, distance):
    bf = cv2.BFMatcher(normType=distance)

    # For each image descriptor, find best k matches among query descriptors
    matches = bf.knnMatch(u, v, k=2)
    return keypoints_based_similarity(lowe_filter(matches))


def flann_matching(u, v, distance):
    index = dict(algorithm=0, trees=5)
    search = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index, search)

    # For each image descriptor, find best k matches among query descriptors
    matches = flann.knnMatch(u, v, k=2)
    return keypoints_based_similarity(lowe_filter(matches))


DISTANCE_METRICS = {
    'l1': cv2.NORM_L1,
    'l2': cv2.NORM_L2,
    'hamming': cv2.NORM_HAMMING,
    'hamming2': cv2.NORM_HAMMING2
}

LOCAL_DESCRIPTORS = {
    'bruteforce': bruteforce_matching,
    'flann': flann_matching,
}

def match_keypoints_descriptors(u, v, method="flann", distance="l2"):
    if u is None or v is None or len(u) == 0 or len(v) == 0:
        return 0.0
    return LOCAL_DESCRIPTORS[method](u.astype(np.float32), v.astype(np.float32), DISTANCE_METRICS[distance])