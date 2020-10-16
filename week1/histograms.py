import cv2
import numpy as np
import os

OPENCV_COLOR_SPACES = {
    "RGB": cv2.COLOR_BGR2RGB,
    "LAB": cv2.COLOR_BGR2LAB,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
    "HSV" : cv2.COLOR_BGR2HSV
}

def _descriptor(image:np.ndarray, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract histogram features from an image
    
    Args:
        image: (H x W x C) 3D  BGR image array of type np.uint8
        
        mask: a mask is a uint8 image with the same shape as our 
        original image, where pixels with a value of zero are 
        ignored and pixels with a value greater than zero are 
        included in the histogram computation. Using masks allow 
        us to only compute a histogram for a particular region of an image
        
    Returns: 
        1D  array of type np.float32 containing histogram
        feautures of image. The length of the array depends on
        the number of bins used for histogram and also on the
        color space used (1-channel vs 3-channel)
    """
    pass

def gray_historam(image:np.ndarray, bins:int=256, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract histogram from grascale version of image
    
    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)
        
    Returns: 
        1D  array of type np.float32 containing histogram
        feautures of image
    """
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image],[0], mask, [bins], [0,256])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()


def rgb_histogram_1d(image:np.ndarray, bins:int=256, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract 1D histogram from BGR image color space by stacking histograms
    from each channel into a single vector
    
    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)
        
    Returns: 
        1D  array of type np.float32 containing histogram
        feautures of image
    """
    channels = cv2.split(image)
    features = []
    
    for channel in channels:
        hist = cv2.calcHist([channel],[0], mask, [bins], [0,256])
        hist = cv2.normalize(hist, hist)
        features.extend(hist)
    # the features from each channel are combined into a single vector here
    return np.stack(features).flatten()

def rgb_histogram_3d(image:np.ndarray, bins:int=8, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract 3D histogram from BGR image color space
    
    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)
        
    Returns: 
        3D histogram features flattened into a 
        1D array of type np.float32
    """
    hist = cv2.calcHist([image], [0, 1, 2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()

def hsv_histogram_1d(image:np.ndarray, bins:int=256, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract 1D histogram from HSV image color space by stacking histograms
    from each channel into a single vector. First channel of HSV takes
    values from (0,180) and the rest 2 take values from (0,255)
    
    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)
        
    Returns: 
        1D  array of type np.float32 containing histogram
        feautures of HSV image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channels = cv2.split(image)
    features = []
    # the brightness channel doesn't get many bins as almost all of the features 
    # were clicked in similar lighting/brightness
    for channel, bin_val, max_val in zip(channels,[int(180/(256/bins)), bins, int(bins/8)],[180,256,256]):
        hist = cv2.calcHist([channel],[0], mask, [bin_val], [0, max_val])
        hist = cv2.normalize(hist, hist)
        features.extend(hist)
    # the features from each channel are combined into a single vector here
    return np.stack(features).flatten()

def hsv_histogram_3d(image:np.ndarray, bins:int=8, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract 3D histogram from HSV image color space
    
    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)
        
    Returns: 
        3D histogram features of HSV image flattened into a 
        1D array of type np.float32
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Focus more on the first channel which is hue, and less on the last channel which is value/brightness
    # Hence first channel gets twice the number of bins, and last channel gets half the number of bins
    # to compensate for keeping the lenght of the feature vector same
    hist = cv2.calcHist([image], [0, 1, 2], mask, [4*bins, bins, int(bins/2)], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()

def lab_histogram_1d(image:np.ndarray, bins:int=256, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract 1D histogram from Lab image color space by stacking histograms
    from each channel into a single vector
    
    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)
        
    Returns: 
        1D  array of type np.float32 containing histogram
        feautures of Lab image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    channels = cv2.split(image)
    features = []
    
    for channel in channels:
        hist = cv2.calcHist([channel],[0], mask, [bins], [0,256])
        hist = cv2.normalize(hist, hist)
        features.extend(hist)
    # the features from each channel are combined into a single vector here
    return np.stack(features).flatten()

def lab_histogram_3d(image:np.ndarray, bins:int=8, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract 3D histogram from Lab image color space
    
    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)
        
    Returns: 
        3D histogram features flattened into a 
        1D array of type np.float32
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # preferential number of bins for each channel based on experimental results
    hist = cv2.calcHist([image], [0, 1, 2], mask, [int(bins/4), 3*bins, 3*bins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()

def ycrcb_histogram_1d(image:np.ndarray, bins:int=256, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract 1D histogram from YCrCb image color space by stacking histograms
    from each channel into a single vector
    
    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)
        
    Returns: 
        1D  array of type np.float32 containing histogram
        feautures of YCrCb image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(image)
    features = []
    
    for channel in channels:
        hist = cv2.calcHist([channel],[0], mask, [bins], [0,256])
        hist = cv2.normalize(hist, hist)
        features.extend(hist)
    # the features from each channel are combined into a single vector here
    return np.stack(features).flatten()

def ycrcb_histogram_3d(image:np.ndarray, bins:int=8, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract 3D histogram from YCrCb image color space.
    
    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)
        
    Returns: 
        3D histogram features flattened into a 
        1D array of type np.float32
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # preferential number of bins for each channel based on experimental results
    hist = cv2.calcHist([image], [0, 1, 2], mask, [int(bins/4), 3*bins,3*bins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()

DESCRIPTORS = {
    "gray_histogram":gray_historam,
    "rgb_histogram_1d":rgb_histogram_1d,
    "rgb_histogram_3d":rgb_histogram_3d,
    "hsv_histogram_1d":hsv_histogram_1d,
    "hsv_histogram_3d":hsv_histogram_3d,
    "lab_histogram_1d":lab_histogram_1d,
    "lab_histogram_3d":lab_histogram_3d,
    "ycrcb_histogram_1d":ycrcb_histogram_1d,
    "ycrcb_histogram_3d":ycrcb_histogram_3d
}

def extract_features(image:np.ndarray, descriptor:str, bins:int, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract features from image based on descriptor of choice

    DESCRIPTORS AVAILABLE
    "gray_historam":gray_historam,
    "rgb_histogram_1d":rgb_histogram_1d,
    "rgb_histogram_3d":rgb_histogram_3d,
    "hsv_histogram_1d":hsv_histogram_1d,
    "hsv_histogram_3d":hsv_histogram_3d,
    "lab_histogram_1d":lab_histogram_1d,
    "lab_histogram_3d":lab_histogram_3d,
    "ycrcb_histogram_1d":ycrcb_histogram_1d,
    "ycrcb_histogram_3d":ycrcb_histogram_3d

    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        descriptor: method used to compute features
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)

    Returns: 
        1D  array of type np.float32 containing histogram
        feautures of image
    """
    return DESCRIPTORS[descriptor](image=image,bins=bins,mask=mask)

if __name__ == '__main__':
    img = cv2.imread(os.path.join('images','barca.png'))
    print(f'image shape: {img.shape}')
    for key,bins in zip(DESCRIPTORS,[256,256,8,256,8,256,8,256,8]):
        print(f'descriptor: {key}, feature_length = {extract_features(img, descriptor=key, bins=bins).shape}')