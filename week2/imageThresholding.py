import imutils
import numpy as np
import argparse
import time
import cv2
import glob
import os
from PIL import Image

# Ignore, this is a command to use my xserver through WSL
# export DISPLAY=`grep -oP "(?<=nameserver ).+" /etc/resolv.conf`:0.0


# Resizes a image and maintains aspect ratio
def resize_mantain_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)


def imageThresholding(original):

    image = original
    imgFlipped = cv2.flip(original, 1)
    #image = cv2.bilateralFilter(image, d=1, sigmaColor=0, sigmaSpace=30)
    
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    h, l, s = cv2.split(hls)

    lowThreshVal = 0
    topThreshVal = 255
    h = getSobel(h)
    l = getSobel(l)
    s = getSobel(s)

    verticalKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,10))
    horizontalKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,1))
    denoiseKernel = np.ones((3,3), np.uint8)
    iters = 1

    hue = binarizeAndMorphChannel_twoDirs(h, lowThreshVal, topThreshVal, denoiseKernel, verticalKernel, horizontalKernel, iters)

    lightness = binarizeAndMorphChannel_twoDirs(l, lowThreshVal, topThreshVal, denoiseKernel, verticalKernel, horizontalKernel, iters)

    saturation = binarizeAndMorphChannel_twoDirs(s, lowThreshVal, topThreshVal, denoiseKernel, verticalKernel, horizontalKernel, iters)

    return hue, lightness, saturation

def binarizeAndMorphChannel_twoDirs(img, lowThreshVal, topThreshVal, denoiseKernel, verticalKernel, horizontalKernel, iters):
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, denoiseKernel)

    ret, tmp = cv2.threshold(
        img, lowThreshVal, topThreshVal, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    imgY = morphChannel(tmp, verticalKernel, iters)
    imgX = morphChannel(tmp, horizontalKernel, iters)

    #tmp = cv2.addWeighted(imgX,0.2, imgY, 0.8, 1)
    tmp = cv2.bitwise_and(imgX, imgY)
    ret, result = cv2.threshold(
        tmp, lowThreshVal, topThreshVal, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return result

def binarizeAndMorphChannel(img, lowThreshVal, topThreshVal, denoiseKernel, verticalKernel, horizontalKernel, iters):
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, denoiseKernel)

    ret, tmp = cv2.threshold(
        img, lowThreshVal, topThreshVal, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    result = morphChannel(tmp, verticalKernel, iters)
    return result

def getSobel(inputImg):    

    scale = 1
    delta = 0
    ddepth = cv2.CV_8U    

    lx = cv2.Sobel(inputImg, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    ly = cv2.Sobel(inputImg, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abslx = cv2.convertScaleAbs(lx)
    absly = cv2.convertScaleAbs(ly)

    inputImg=cv2.addWeighted(abslx, 0.2, absly, 0.8, 0)
    inputImg = cv2.bitwise_or(abslx, absly)
    return inputImg

def morphChannel(img, kernel, iters):
    img = cv2.dilate(img, kernel, iterations=iters)
    img = cv2.erode(img, kernel, iterations=iters)

    img = cv2.erode(img, kernel, iterations=iters)
    img = cv2.dilate(img, kernel, iterations=iters)

    return img


if __name__ == '__main__':
    impath = '../resources/qsd1_w2'
    extension = 'jpg'
    images = sorted(glob.glob(os.path.join(impath, '*.'+extension)))

    print(f'number of images in dataset is {len(images)}')
    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)

    i = 0

    for im in images:
        image = cv2.imread(im)
        h, l, s = imageThresholding(image)

        width = 600
        height = 600
        concat_img = np.concatenate((resize_mantain_ratio(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), width, height),
                                     resize_mantain_ratio(h, width, height),
                                     resize_mantain_ratio(l, width, height),
                                     resize_mantain_ratio(s, width, height)), axis=1)


        print("Image {}".format(i))
        cv2.imshow('Image', concat_img)
        cv2.waitKey(0)
        i = i+1
        '''if i > 5:
            break'''

    cv2.destroyAllWindows()
