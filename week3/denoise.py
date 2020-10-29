

import numpy as np
import fastsearch
import cv2
import math
from scipy import signal

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
    
def image_sigma(img):
    # paper: https://www.sciencedirect.com/science/article/abs/pii/S1077314296900600
    height = img.shape[0]
    width = img.shape[1]
    mask = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    convolved = signal.convolve2d(img, mask)
    sigma = np.sum(np.sum(np.absolute(convolved)))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (width-2) * (height-2))

    return sigma


def sigma_grid(img):
    partitions = 16
    grid_image_sigma = 0
    height = img.shape[0]
    width = img.shape[1]
        
    newH = height//partitions
    newW = width//partitions


    for y in range(0,height,newH):
        for x in range(0, width, newW):
            partialImg = img[y:y+newH,x:x+newW]
            grid_image_sigma += image_sigma(partialImg)

    return grid_image_sigma / (partitions * partitions)

  

def get_image_noisy_list(img_path_list):
    img_is_noisy_list = []
    for idx, img_path in enumerate(img_path_list):
        img = cv2.imread(img_path, 0)
        current_img_sigma = round(image_sigma(img), 3)
        img_is_noisy_list.append(current_img_sigma >=6.5)
        # print(f'Image {img_path} has sigma = {current_img_sigma}\t and is \t{"NOISY" if img_is_noisy_list[idx] else "NOT NOISY"}')
    return img_is_noisy_list


def median_blur_denoise(img):
    denoised_img = img

    # denoised_img = cv2.fastNlMeansDenoisingColored(
    #     img
    #     , None
    #     , templateWindowSize=7
    #     , searchWindowSize=21
    #     , h=7
    #     , hColor=21)

    denoised_img = cv2.medianBlur(img, 3)
    return denoised_img

def fast_NL_Means_denoise(img):
    denoised_img = cv2.fastNlMeansDenoisingColored(
        img
        , None
        , templateWindowSize=7
        , searchWindowSize=21
        , h=7
        , hColor=21)

    denoised_img = cv2.medianBlur(img, 3)
    return denoised_img
    


if __name__ == '__main__':
    query_path = '../resources/qsd1_w3'
    img_path_list = fastsearch.get_image_path_list(query_path)
    
    img_is_noisy_list = get_image_noisy_list(img_path_list)
    
    for idx, img_is_nosy in enumerate(img_is_noisy_list):
        if img_is_nosy:
            # print(f'Image {img_path_list[idx]} is  {"NOISY" if img_is_nosy else "NOT NOISY"}')

            original_img = cv2.imread(img_path_list[idx])
            denoised_img = median_blur_denoise(original_img)
            original_img = resize_mantain_ratio(original_img, 600,600)
            denoised_img = resize_mantain_ratio(denoised_img, 600,600)
            concat_img = np.concatenate((original_img, denoised_img), axis=1)
            cv2.imshow("img", concat_img)
            Key = cv2.waitKey(0)
            if (Key == ord('a')):
                break
    
    cv2.destroyAllWindows()

            
