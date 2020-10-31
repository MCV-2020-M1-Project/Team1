

import numpy as np
import fastsearch
import cv2
import math
import os
from scipy import signal, stats
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise

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

  

def get_image_noisy_list(img_path_list, original_sigma_list):
    img_is_noisy_list = []
    for idx, img_path in enumerate(img_path_list):
        img = cv2.imread(img_path, 0)
        current_img_sigma = round(image_sigma(img), 3)
        original_sigma_list.append(current_img_sigma)
        img_is_noisy_list.append(current_img_sigma >=6.5)
        # print(f'Image {img_path} has sigma = {current_img_sigma}\t and is \t{"NOISY" if img_is_noisy_list[idx] else "NOT NOISY"}')
    return img_is_noisy_list


def median_blur_denoise(img):
    
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

def bilateral_denoise(img):
    sigmaColor = 9
    sigmaSpace = 75
    borderType = 75

    denoised_img = cv2.bilateralFilter(img, sigmaColor, sigmaSpace, borderType, dst = None )

    return denoised_img

DENOISING_METHODS = {
    "bilateral_filtering": bilateral_denoise,
    "fast_nl_means": fast_NL_Means_denoise,
    "median_blur": median_blur_denoise,
    # "linear" : linear_denoise
    ####


}   
    
def eval_best_denoise(img, original_sigma):

    best_denoised_image = img
    best_denoised_sigma = original_sigma
    best_denoised_method = "none"
    
    for key,denoise_method in DENOISING_METHODS:

        denoised_img = denoise_method[key](img)
        current_sigma = image_sigma(denoised_img)

        if current_sigma < best_denoised_sigma:
            best_denoised_sigma = current_sigma
            best_denoised_image = denoised_img
            best_denoised_method = denoise_method

    return best_denoised_image, best_denoised_method, best_denoised_method

def eval_PSNR():
    a=0
    return a


if __name__ == '__main__':
    query_path = '../resources/qsd1_w3'
    non_augmented_path = '../resources/qsd1_w3/non_augmented'

    img_path_list = fastsearch.get_image_path_list(query_path)
    non_augmented_img_path_list = fastsearch.get_image_path_list(non_augmented_path)

    original_sigma_list = []    
    img_is_noisy_list = get_image_noisy_list(img_path_list, original_sigma_list)
    print(len(original_sigma_list))
    denoised_img_list = []
    subidx = 0
    
    print(f'psnr_non_augmented_to_augmented\t|\tpsnr_non_augmented_to_denoised\t|\tdelta')
    for idx, img_is_nosy in enumerate(img_is_noisy_list):
        if img_is_nosy:
            # print(f'Image {img_path_list[idx]} is  {"NOISY" if img_is_nosy else "NOT NOISY"}')

            original_img = cv2.imread(img_path_list[idx])
            non_augmented_img = cv2.imread(non_augmented_img_path_list[idx])

            denoised_img = median_blur_denoise(original_img)
            denoised_img = eval_best_denoise(original_img, original_sigma_list)
            
            psnr_non_augmented_to_augmented = peak_signal_noise_ratio(non_augmented_img, original_img)
            psnr_non_augmented_to_denoised = peak_signal_noise_ratio(non_augmented_img, denoised_img)
            
            #cv2.imwrite(f'../resources/qsd2_w3_denoised/denoised{idx}.jpg', denoised_img) 

            denoised_img_list.append(denoised_img)
            subidx = subidx + 1

            ##################################################################
            ######################  only for imshow ##########################
            ##################################################################

            original_img = resize_mantain_ratio(original_img, 600,600)
            denoised_img = resize_mantain_ratio(denoised_img, 600,600)
            concat_img = np.concatenate((original_img, denoised_img), axis=1)
            # dct = cv2.cvtColor(concat_img, cv2.COLOR_BGR2GRAY)
            # dct = np.float32(dct)/255.0
            # dct = cv2.dct(dct)

            # print(f'psnr_non_augmented_to_augmented: {psnr_non_augmented_to_augmented}\tpsnr_non_augmented_to_denoised: {psnr_non_augmented_to_denoised}')
            delta = psnr_non_augmented_to_augmented-psnr_non_augmented_to_denoised
            print(f'{psnr_non_augmented_to_augmented:.2f}\t\t|\t\t{psnr_non_augmented_to_denoised:.2f}\t\t|\t\t{delta:.2f}')
            cv2.imshow("img", concat_img)
            Key = cv2.waitKey(0)
            if ((Key == ord('a')) | (Key == ord('A'))| (Key == 27)):
                break
            ##################################################################
            #####################  only for imshow ###########################
            ##################################################################

    
    
    cv2.destroyAllWindows()

            
