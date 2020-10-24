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
    #image = cv2.bilateralFilter(image, d=0, sigmaColor=2, sigmaSpace=20)
    
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    h, l, s = cv2.split(hls)

    lowThreshVal = 25
    topThreshVal = 255


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    h = cv2.morphologyEx(h, cv2.MORPH_OPEN, kernel)

    ret, hue = cv2.threshold(
        h, lowThreshVal, topThreshVal, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    hue = cv2.dilate(hue, kernel, iterations=1)
    hue = cv2.erode(hue, kernel, iterations=1)

    hue = cv2.erode(hue, kernel, iterations=1)
    hue = cv2.dilate(hue, kernel, iterations=1)

    l = cv2.morphologyEx(l, cv2.MORPH_OPEN, kernel)
    ret, lightness = cv2.threshold(
        l, lowThreshVal, topThreshVal, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    lightness = cv2.dilate(lightness, kernel, iterations=1)
    lightness = cv2.erode(lightness, kernel, iterations=1)

    lightness = cv2.erode(lightness, kernel, iterations=1)
    lightness = cv2.dilate(lightness, kernel, iterations=1)

    s = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
    ret, saturation = cv2.threshold(
        s, lowThreshVal, topThreshVal, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    saturation = cv2.dilate(saturation, kernel, iterations=1)
    saturation = cv2.erode(saturation, kernel, iterations=1)

    saturation = cv2.erode(saturation, kernel, iterations=1)
    saturation = cv2.dilate(saturation, kernel, iterations=1)
    return hue, lightness, saturation


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
