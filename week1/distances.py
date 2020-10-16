import cv2
import numpy as np

def _distance(u:np.ndarray, v:np.ndarray) -> float:
    """
    Dummy distance function based on some similarity measure.
    The higher the distance, the lower the similarity.

    Args:
        u: 1D array of type np.float32 containing image descriptors
        v: 1D array of type np.float32 containing image descriptors
    
    Returns : distance between input descriptor vectors
    """
    pass

def cosine(u:np.ndarray, v:np.ndarray) -> float:
    """
    Compare histograms based on cosine similarity
    Args:
        u: 1D array of type np.float32 containing image descriptors
        v: 1D array of type np.float32 containing image descriptors
    
    Returns : distance between input descriptor vectors
    """
    # since cosine distance measures similarity, we subtract it from 1 to get the distance
    return 1 - np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))

def manhattan(u:np.ndarray, v:np.ndarray) -> float:
    """
    Compare histograms based on the euclidean distance (L2 norm)

    Args:
        u: 1D array of type np.float32 containing image descriptors
        v: 1D array of type np.float32 containing image descriptors
    
    Returns : distance between input descriptor vectors
    """
    return np.linalg.norm(u-v,1)

def euclidean(u:np.ndarray, v:np.ndarray) -> float:
    """
    Compare histograms based on the euclidean distance (L2 norm)

    Args:
        u: 1D array of type np.float32 containing image descriptors
        v: 1D array of type np.float32 containing image descriptors
    
    Returns : distance between input descriptor vectors
    """
    return np.linalg.norm(u-v)

def intersect(u:np.ndarray, v:np.ndarray) -> float:
    """
    Compare histograms based on their intersection

    Args:
        u: 1D array of type np.float32 containing image descriptors
        v: 1D array of type np.float32 containing image descriptors
    
    Returns : distance between input descriptor vectors
    """
    # since intersection measures similarity, we use its negative to get the distance
    return -cv2.compareHist(u, v, cv2.HISTCMP_INTERSECT)


def kl_div(u:np.ndarray, v:np.ndarray) -> float:
    """
    Compare histograms based on the Kullback-Leibler divergence.

    Args:
        u: 1D array of type np.float32 containing image descriptors
        v: 1D array of type np.float32 containing image descriptors
    
    Returns : distance between input descriptor vectors
    """
    return cv2.compareHist(u, v, cv2.HISTCMP_KL_DIV)

def bhattacharyya(u:np.ndarray, v:np.ndarray) -> float:
    """
    Compare histograms based on the Bhattacharya distance

    Args:
        u: 1D array of type np.float32 containing image descriptors
        v: 1D array of type np.float32 containing image descriptors
    
    Returns : distance between input descriptor vectors
    """
    return cv2.compareHist(u, v, cv2.HISTCMP_BHATTACHARYYA)

def hellinger(u:np.ndarray, v:np.ndarray) -> float:
    """
    Compare histograms based on the Hellinger distance

    Args:
        u: 1D array of type np.float32 containing image descriptors
        v: 1D array of type np.float32 containing image descriptors
    
    Returns : distance between input descriptor vectors
    """
    return cv2.compareHist(u, v, cv2.HISTCMP_HELLINGER)

def chisqr(u:np.ndarray, v:np.ndarray) -> float:
    """
    Compare histograms based on Chi-Square Test

    Args:
        u: 1D array of type np.float32 containing image descriptors
        v: 1D array of type np.float32 containing image descriptors
    
    Returns : distance between input descriptor vectors
    """
    return cv2.compareHist(u, v, cv2.HISTCMP_CHISQR)

def correl(u:np.ndarray, v:np.ndarray) -> float:
    """
    Compare histograms based on correlation

    Args:
        u: 1D array of type np.float32 containing image descriptors
        v: 1D array of type np.float32 containing image descriptors
    
    Returns : distance between input descriptor vectors
    """
    # since correlation measures similarity, we subtract it from 1 to get the distance
    return 1 - cv2.compareHist(u, v, cv2.HISTCMP_CORREL)


DISTANCE_MEASURES = {
    "cosine": cosine,
    "manhattan": manhattan,
    "euclidean": euclidean,
    "intersect":intersect,
    "kl_div":kl_div,
    "bhattacharyya":bhattacharyya,
    "hellinger":hellinger,
    "chisqr":chisqr,
    "correl":correl
}

def compute_distance(u:np.ndarray, v:np.ndarray, metric:str) -> float:
    """
    Calculate distance between input histograms based on 
    similarity metric of choice

    METRICS AVAILABLE
    "cosine": cosine,
    "manhattan": manhattan,
    "euclidean": euclidean,
    "intersect":intersect,
    "kl_div":kl_div,
    "bhattacharyya":bhattacharyya,
    "hellinger":hellinger,
    "chisqr":chisqr,
    "correl":correl

    Args:
        u: 1D array of type np.float32 containing normalized image descriptors
        v: 1D array of type np.float32 containing normalized image descriptors
        metric: distance function of choice
        
    Returns : distance between input descriptor vectors
    """
    # OpenCV Functions only work with float32 numbers
    assert u.dtype == np.float32
    assert v.dtype == np.float32
    
    return DISTANCE_MEASURES[metric](u,v)

if __name__ == '__main__':
    u = np.array([1,2,3,4],dtype=np.float32)
    v = np.array([4,3,2,1],dtype=np.float32)
    for key in DISTANCE_MEASURES:
        print(f'measure: {key}, distance = {compute_distance(u, v, key)}')
