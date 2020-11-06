import cv2
import numpy as np
import os
from functools import partial
from skimage.feature import local_binary_pattern, hog
import multiprocessing.dummy as mp

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
    print(mask is None)
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

def block_descriptor(image:np.ndarray, descriptor_func=rgb_histogram_1d, bins:int=8, mask:np.ndarray=None, num_blocks:int=1) -> np.ndarray:
    """
    Extract descriptors after dividing image in non-overlapping blocks,
    computing histograms for each block and then concatenating them

    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        descriptor_func: descriptorf function to extract histogram
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)
        num_blocks: number of blocks to divide images into

    Returns:
        Histogram features flattened into a 
        1D array of type np.float32
    """
    #h,w = image.shape[:2]
    if mask is not None:
        x,y,w,h = cv2.boundingRect(mask)
        block_h = int(np.ceil(h / num_blocks))
        block_w = int(np.ceil(w / num_blocks))
    else:
        x,y = 0,0
        h,w = image.shape[:2]
        block_h = int(np.ceil(h / num_blocks))
        block_w = int(np.ceil(w / num_blocks))
        
    features = []
    #for i in range(0, h, block_h):
    for i in range(y, y+h, block_h):
        #for j in range(0, w, block_w):
        for j in range(x, x+w, block_w):
            image_block = image[i:i+block_h, j:j+block_w]
            if mask is not None:
                mask_block = mask[i:i+block_h, j:j+block_w]
            else:
                mask_block = None
            block_feature =  descriptor_func(image=image_block, bins=bins, mask=mask_block)
            features.extend(block_feature)
    return np.stack(features).flatten()

def pyramid_descriptor(image:np.ndarray, descriptor_func=rgb_histogram_1d, bins:int=8, mask:np.ndarray=None, max_level:int=1) -> np.ndarray:
    """
    Compute block histogram features at different levels

    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        descriptor_func: descriptorf function to extract histogram
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)
        max_level: number of levels to use for histogram generation

    Returns:
        Histogram features flattened into a 
        1D array of type np.float32
    """
    features = []
    for level in range(1, max_level+1):
        num_blocks = 4 ** (level-1)
        features.extend(block_descriptor(image, descriptor_func, bins, mask, num_blocks))
    return np.stack(features).flatten()  

def lbp_histogram_uniform(image:np.ndarray, points:int=8, radius:float=2.0, bins:int=8, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract LBP descriptors after dividing image in non-overlapping blocks,
    computing histograms for each block and then concatenating them

    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        points: number of circularly symmetric neighbour set points (quantization of the angular space)
        radius: radius of circle (spatial resolution of the operator)
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)

    Returns:
        Histogram features flattened into a 
        1D array of type np.float32
        
    EXTRA DOCUMENTATION FOR PERSONAL USE
        For the method DEFAULT and points 8: 
         max: 255.0, min: 0.0, unique_values: 212
        For the method ROR: 
         max: 255.0, min: 0.0, unique_values: 34
        For the method UNIFORM: 
         max: 9.0, min: 0.0, unique_values: 10
        For the method NRI_UNIFORM: 
         max: 58.0, min: 0.0, unique_values: 59
    """    
    # image --> grayscale --> lbp --> histogram
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = (local_binary_pattern(image, points, radius, method="uniform")).astype(np.uint8)
    bins = points + 2
    hist = cv2.calcHist([image],[0], mask, [bins], [0, bins])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()

def lbp_histogram_default(image:np.ndarray, points:int=8, radius:float=2.0, bins:int=16, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract LBP descriptors after dividing image in non-overlapping blocks,
    computing histograms for each block and then concatenating them

    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        points: number of circularly symmetric neighbour set points (quantization of the angular space)
        radius: radius of circle (spatial resolution of the operator)
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)

    Returns:
        Histogram features flattened into a 
        1D array of type np.float32
    """
    # image --> grayscale --> lbp --> histogram
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    image = (local_binary_pattern(image, points, radius, method="default")).astype(np.uint8)
    hist = cv2.calcHist([image],[0], mask, [bins], [0, 256])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()

def dct_coefficients(image:np.ndarray, bins:int=8, mask:np.ndarray=None, num_coeff:int=4) -> np.ndarray:
    # image --> grayscale --> DCT --> get top N coefficients using zig-zag scan
    """
    Extract DCT coefficients from image. This descriptor will be clubbed with a block descriptor

    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        num_coeff: number of coefficents in dct_block to use through zig-zag scan
        bins: N.A. here, but present to make our api compatible with the function
        mask: check _descriptor(first function in file)

    Returns:
        DCT features flattened into a 
        1D array of type np.float32
    """    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is not None:
        image = cv2.bitwise_and(image, image, mask=mask)
        
    block_dct = cv2.dct(np.float32(image)/255.0)

    def _compute_zig_zag(a):
        return np.concatenate([np.diagonal(a[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-a.shape[0], a.shape[0])])
    
    features = _compute_zig_zag(block_dct[:6,:6])[:num_coeff]
    return features

def hog_(image:np.ndarray, bins:int=8, ppc:int = 16, cpb:int=3, mask:np.ndarray=None) -> np.ndarray:
    
    if mask is not None:
        x,y,w,h = cv2.boundingRect(mask)
        image = image[x:x+w,y:y+h]
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    return np.float32(hog(image, orientations=9, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb),
            block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True,
            multichannel=True))

DESCRIPTORS = {
    "gray_histogram":gray_historam,
    "rgb_histogram_1d":rgb_histogram_1d,
    "rgb_histogram_3d":rgb_histogram_3d,
    "hsv_histogram_1d":hsv_histogram_1d,
    "hsv_histogram_3d":hsv_histogram_3d,
    "lab_histogram_1d":lab_histogram_1d,
    "lab_histogram_3d":lab_histogram_3d,
    "ycrcb_histogram_1d":ycrcb_histogram_1d,
    "ycrcb_histogram_3d":ycrcb_histogram_3d,
    "rgb_histogram_3d_blocks": partial(block_descriptor, descriptor_func=rgb_histogram_3d, num_blocks = 8),
    "lab_histogram_3d_blocks": partial(block_descriptor, descriptor_func=lab_histogram_3d, num_blocks = 8),
    "rgb_histogram_3d_pyramid": partial(pyramid_descriptor, descriptor_func=rgb_histogram_3d, max_level = 1),
    "lab_histogram_3d_pyramid": partial(pyramid_descriptor, descriptor_func=lab_histogram_3d, max_level = 1)
    }

TEXTURES = {
    "lbp_histogram_blocks": partial(block_descriptor, descriptor_func=lbp_histogram_default, num_blocks = 8),
    "dct_blocks": partial(block_descriptor, descriptor_func=dct_coefficients, num_blocks = 8),
    "hog":hog_
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

def extract_textures(image:np.ndarray, descriptor:str, bins:int, mask:np.ndarray=None) -> np.ndarray:
    """
    Extract features from image based on texture descriptor of choice

    DESCRIPTORS AVAILABLE
    "lbp_histogram_blocks"

    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        descriptor: method used to compute features
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)

    Returns: 
        1D  array of type np.float32 containing histogram
        feautures of image
    """
    return TEXTURES[descriptor](image=image,bins=bins,mask=mask)

if __name__ == '__main__':
    img = cv2.imread(os.path.join('images','barca.png'))
    print(f'image shape: {img.shape}')
    print('COLOR DESCRIPTORS')
    for key,bins in zip(DESCRIPTORS,[256,256,8,256,8,256,8,256,8,8,8,8,8]):
        print(f'descriptor: {key}, feature_length = {extract_features(img, descriptor=key, bins=bins).shape}')

    print('TEXTURE DESCRIPTORS')
    for key,bins in zip(TEXTURES,[8,,16]):
        print(f'descriptor: {key}, feature_length = {extract_textures(img, descriptor=key, bins=bins).shape}')