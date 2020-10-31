import argparse
import math
import os
import sys

import cv2
import numpy as np
import skimage.restoration
from scipy import signal, stats
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise

import fastsearch

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

    for y in range(0, height, newH):
        for x in range(0, width, newW):
            partialImg = img[y:y+newH, x:x+newW]
            grid_image_sigma += image_sigma(partialImg)

    return grid_image_sigma / (partitions * partitions)


def get_image_noisy_list(img_path_list, original_sigma_list):
    img_is_noisy_list = []
    for idx, img_path in enumerate(img_path_list):
        img = cv2.imread(img_path, 0)
        # current_img_sigma = round(image_sigma(img), 3)
        current_img_sigma = skimage.restoration.estimate_sigma(
            img, multichannel=False, average_sigmas=False)
        original_sigma_list.append(current_img_sigma)
        img_is_noisy_list.append(current_img_sigma >= 6.5)
        # print(f'Image {idx} has sigma = {current_img_sigma:.4f}\t and is \t{"NOISY" if img_is_noisy_list[idx] else "NOT NOISY"}')
    return img_is_noisy_list


def median_blur_denoise(img):

    denoised_img = cv2.medianBlur(img, 3)
    
    # cv2.imshow("img", denoised_img)
    # Key = cv2.waitKey(0)

    return denoised_img


def fast_NL_Means_denoise(img):
    denoised_img = cv2.fastNlMeansDenoisingColored(
        img, None, templateWindowSize=7, searchWindowSize=21, h=7, hColor=21)
    
    # cv2.imshow("img", denoised_img)
    # Key = cv2.waitKey(0)

    return denoised_img


def bilateral_denoise(img):
    sigmaColor = 100
    sigmaSpace = 5000
    d = 25
    borderType = cv2.BORDER_REPLICATE

    denoised_img = cv2.bilateralFilter(
        img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace, borderType=borderType, dst=None)

    # cv2.imshow("img", denoised_img)
    # Key = cv2.waitKey(0)

    return denoised_img


def bayesshrink_wavelet(img):
    # denoised_img = median_blur_denoise(img)
    sigma_est = skimage.restoration.estimate_sigma(
        img, multichannel=True, average_sigmas=True)

    if ((original_img.shape[0]*original_img.shape[1]) > 600000):
        sigma_est = sigma_est/4
    else:
        sigma_est = sigma_est / 8

    denoised_img = skimage.restoration.denoise_wavelet(img, multichannel=True, convert2ycbcr=True,
                                                       method='BayesShrink', mode='soft',
                                                       rescale_sigma=False, wavelet='db1', sigma=sigma_est)
    denoised_img = normalize_img_from_skimage(denoised_img)
    
    # cv2.imshow("img", denoised_img)
    # Key = cv2.waitKey(0)

    return denoised_img


def visushrink_wavelet(img):
    # denoised_img = median_blur_denoise(img)
    sigma_est = skimage.restoration.estimate_sigma(
        img, multichannel=True, average_sigmas=True)

    if ((original_img.shape[0]*original_img.shape[1]) > 600000):
        sigma_est = sigma_est/4
    else:
        sigma_est = sigma_est / 8

    denoised_img = skimage.restoration.denoise_wavelet(img, multichannel=True, convert2ycbcr=True,
                                                       method='VisuShrink', mode='soft',
                                                       rescale_sigma=False, wavelet='db1', sigma=sigma_est)
    denoised_img = normalize_img_from_skimage(denoised_img)
    
    # cv2.imshow("img", denoised_img)
    # Key = cv2.waitKey(0)

    return denoised_img


def tv_chambolle_denoise(img):

    denoised_img = skimage.restoration.denoise_tv_chambolle(
        img, weight=0.1, multichannel=True, n_iter_max=300)
    denoised_img = normalize_img_from_skimage(denoised_img)

    return denoised_img


def normalize_img_from_skimage(img):
    normalized_img = cv2.normalize(
        src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized_img


DENOISING_METHODS = {
    "bilateral_filtering": bilateral_denoise,
    "fast_nl_means": fast_NL_Means_denoise,
    "median_blur": median_blur_denoise,
    "bayesshrink_wavelet": bayesshrink_wavelet,
    "visushrink_wavelet": visushrink_wavelet,
    "tv_chambolle_denoise": tv_chambolle_denoise,


    # "linear" : linear_denoise
    ####


}


def eval_best_denoise(writer, imgID, img, non_augmented_img, original_sigma, save_all_images):

    best_denoised_image = img
    best_denoised_sigma = original_sigma
    best_denoised_method = "none"
    psnr_non_augmented_to_augmented = peak_signal_noise_ratio(
        non_augmented_img, original_img)
    best_psnr = psnr_non_augmented_to_augmented

    for denoise_method in DENOISING_METHODS:

        denoised_img = DENOISING_METHODS[denoise_method](img)

        # print(denoised_img.shape)
        # cv2.imshow("img", denoised_img)
        # Key = cv2.waitKey(0)

        current_sigma = image_sigma(cv2.cvtColor(
            denoised_img, cv2.COLOR_BGR2GRAY, dst=None))

        cv2.imwrite(f"outputs/partials/{denoise_method}/{str(idx).zfill(5)}.jpg", denoised_img)
        # current_sigma = estimate_sigma(denoised_img, multichannel = True, average_sigmas=True)
        # print(f'{denoise_method} |\t\t{current_sigma}')
        
        psnr_non_augmented_to_denoised = peak_signal_noise_ratio(
            non_augmented_img, denoised_img)

        result_line = f"{idx} | {psnr_non_augmented_to_augmented:.6f} | {psnr_non_augmented_to_denoised:.6f} | {current_sigma} | {denoise_method}\n"

        writer.write(result_line)

        if current_sigma <= best_denoised_sigma:
            best_denoised_sigma = current_sigma
            best_denoised_image = denoised_img
            best_denoised_method = denoise_method

    return best_denoised_image, best_denoised_sigma, best_denoised_method


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Denoising evaluator')

    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Output stats for the best method per image")

    parser.add_argument(
        "--show_images", "-si", action="store_true",
        help="Show the best denoised image, the noisy image, and the non-augmented image")
        
    parser.add_argument(
        "--save_resulting_images", "-sr", action="store_true",
        help="Save only the best denoised images")

    parser.add_argument(
        "--save_all_images", "-sa", action="store_true",
        help="Save ALL OF THE DENOISED IMAGES. CAREFUL")

    parser.add_argument(
        "--query_path", "-q", default="../resources/qsd1_w3",
        help="path to query dataset. Example input: 'data/qsd1_w3'")

    parser.add_argument(
        "--non_augmented_path", "-na", default="../resources/qsd1_w3/non_augmented",
        help="path to clean images. Example input: 'data/qsd1_w3/non_augmented'")

    args = parser.parse_args(args)
    return args


if __name__ == '__main__':

    args = parse_args()
    print(args)
    query_path = args.query_path
    non_augmented_path = args.non_augmented_path
    verbose = args.verbose
    show_images = args.show_images
    save_all_images = args.save_all_images
    save_resulting_images = args.save_resulting_images

    img_path_list = fastsearch.get_image_path_list(query_path)
    non_augmented_img_path_list = fastsearch.get_image_path_list(
        non_augmented_path)

    original_sigma_list = []
    img_is_noisy_list = get_image_noisy_list(
        img_path_list, original_sigma_list)

    # kdx = 0
    # for img_is_nosy in img_is_noisy_list:
    #     if img_is_nosy == True:
    #         kdx = kdx+1
    # print(kdx)

    denoised_img_list = []
    subidx = 0

    if save_all_images | save_resulting_images:
        if not os.path.exists("outputs"):
            os.mkdir("outputs")

    if save_resulting_images:
        resulting_images_dir = "outputs/results"
        print(f"Saving resulting images to path: {resulting_images_dir}")
        if not os.path.exists(resulting_images_dir):
            os.makedirs(resulting_images_dir)

    if save_all_images:
        if not os.path.exists("outputs/partials"):
            os.makedirs("outputs/partials")
        
        partial_imgs_dir = "outputs/partials"
        print(f"Saving partial images to path: {partial_imgs_dir}/denoising_method/")
        
        denoise_method_folders = os.path.join("outputs/", "partials")
        for denoise_method in DENOISING_METHODS:
            denoising_method_subfolder = os.path.join(denoise_method_folders, denoise_method)
            if not os.path.exists(denoising_method_subfolder):
                os.makedirs(denoising_method_subfolder)
        

    #### initialize data dump file ########
    denoise_data_dump_file = "denoise_data_dump.txt"
    denoise_data_dump_file_writer = open(denoise_data_dump_file, 'w')
    results_header = "Index | psnr_non_augmented_to_augmented | psnr_non_augmented_to_denoised | denoised_sigma | denoise_method\n"
    denoise_data_dump_file_writer.write(results_header)
    #### initialize data dump file ########

    if verbose:
        print(f'psnr_non_augmented_to_augmented | psnr_non_augmented_to_denoised | delta | denoised_sigma | denoise_method')

    for idx, img_is_nosy in enumerate(img_is_noisy_list):
        if img_is_nosy:
            # print(f'Image {img_path_list[idx]} is  {"NOISY" if img_is_nosy else "NOT NOISY"}')

            original_img = cv2.imread(img_path_list[idx])
            non_augmented_img = cv2.imread(non_augmented_img_path_list[idx])

            # denoised_img = median_blur_denoise(original_img)
            denoised_img, denoised_sigma, denoise_method = eval_best_denoise(
                denoise_data_dump_file_writer, idx, original_img, non_augmented_img, original_sigma_list[idx], save_all_images)

            #cv2.imwrite(f'../resources/qsd2_w3_denoised/denoised{idx}.jpg', denoised_img)

            denoised_img_list.append(denoised_img)
            subidx = subidx + 1

            ##################################################################
            ######################  only for imshow ##########################
            ##################################################################

            if verbose:
                psnr_non_augmented_to_augmented = peak_signal_noise_ratio(
                    non_augmented_img, original_img)
                psnr_non_augmented_to_denoised = peak_signal_noise_ratio(
                    non_augmented_img, denoised_img)
                # print(f'psnr_non_augmented_to_augmented: {psnr_non_augmented_to_augmented}\tpsnr_non_augmented_to_denoised: {psnr_non_augmented_to_denoised}')
                delta = psnr_non_augmented_to_augmented-psnr_non_augmented_to_denoised

                print(f'{psnr_non_augmented_to_augmented:.3f}|\t{psnr_non_augmented_to_denoised:.3f}|\t{delta:.3f}|\t{denoised_sigma:.3f}|\t{denoise_method}')

            if show_images:
                original_img = resize_mantain_ratio(original_img, 600, 600)
                denoised_img = resize_mantain_ratio(denoised_img, 600, 600)
                non_augmented_img = resize_mantain_ratio(non_augmented_img, 600, 600)
                concat_img = np.concatenate((denoised_img, original_img, non_augmented_img), axis=1)
                dct = cv2.cvtColor(concat_img, cv2.COLOR_BGR2GRAY)
                dct = np.float32(dct)/255.0
                dct = cv2.dct(dct)
                cv2.imshow("img", concat_img)
                Key = cv2.waitKey(0)
                if ((Key == ord('a')) | (Key == ord('A'))| (Key == 27)):
                    break

            ##################################################################
            #####################  only for imshow ###########################
            ##################################################################

    cv2.destroyAllWindows()
