import argparse
import sys
import cv2
import glob, os, pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
from functools import partial
import multiprocessing.dummy as mp

from distances import compute_distance
from histograms import extract_features
from evaluation import mapk
from masks import extract_paintings_from_mask

def get_image_path_list(data_path:str, extension:str='jpg') -> List[Path]:
    """
    Used for indexing all images in a folder with the given
    extension
    
    Args:
        data_path: path to folder containing the images
        extension: type of image, can be 'jpg' or 'png'
    
    Returns: 
        a list of paths to all the images in the folder
        with the given extension
    """
    return sorted(glob.glob(os.path.join(data_path,'*.'+extension)))

def path2img(path:Path, rgb:bool=False) -> np.ndarray:
    """
    Convert image to numpy array from path
    
    Args: 
        path: path to image
        rgb: choose to load image in 'RGB' format
            instead of default 'BGR' of OpenCV

    Returns: 
        a numpy array image from specified path
    """
    if rgb:
        return np.array(Image.open(path))
    else:
        return cv2.imread(path)

def path2mask(path:Path) -> np.ndarray:
    """
    Convert image to numpy array from path
    
    Args: 
        path: path to image

    Returns: 
        a numpy array image from specified path
    """
    mask = cv2.imread(path)
    gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    return gray_mask

def plot_results(image_path_list: List[Path]) -> None:
    """
    Show top k-images fetched from museum dataset
    that are most similar to the query images
    """
    fig,axes = plt.subplots(3,2, figsize=(16,8))
    for ax, path, title in zip(axes.flatten(), image_path_list ,['query image','retrieval1','retrieval2','retrieval3','retrieval4','retrieval5']):
        ax.imshow(path2img(path, rgb=True ))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def get_ground_truth(path:str=None) -> List[List[int]]:
    """
    Retrieve pickle file containing ground truths
    of query images

    Args:
        path: path to the pickle file

    Returns:
        the contents of the pickle file
        
    """
    gt_file_list = glob.glob(os.path.join(path,'*gt*.pkl'))
    with open(gt_file_list[0], 'rb') as f:
        query_gt = pickle.load(f)
    return query_gt

def search(img_path:Path=None, descriptor:str='rgb_histogram_1d', metric:str="euclidean", bins:int=64, k:int=10) -> List[int]:
    """
    Searches the reference dataset for most similar images to the
    query image

    Args:
        img_path: path to the query image
        descriptor: method used to compute features
        metric: similarity measure to compare images
        bins: number of bins to use for histogram
        k: to return top k images that are most similar

    Returns:
        a list of indices of top k images in reference dataset
        that are most similar to query image
    """
    
    # define particular descriptors and distance metrics to be used
    # so that you don't have to input the function arguments every 
    #time you call the function
    extract_features_func = partial(extract_features, descriptor=descriptor,bins=bins)
    distance_func = partial(compute_distance, metric=metric)
    
    def _extract_features_from_path(_path):
        return extract_features_func(path2img(_path))

    query_feature = _extract_features_from_path(img_path)

    with mp.Pool(processes=20) as p:
        image_descriptors = p.map(_extract_features_from_path, [ref_path for ref_path in museum_list])
        dist = p.starmap(distance_func, [(query_feature, ref_feature) for ref_feature in image_descriptors])
    
    nearest_indices = np.argsort(dist)[:k]
    #print(f'{k} Most similar images in the reference dataset are {nearest_indices}')
    result_list = [index for index in nearest_indices]
    return result_list


def search_batch(query_list:List[Path]=None, mask_list:List[Path]=None ,descriptor:str='rgb_histogram_1d', metric:str="euclidean", bins:int=64, k:int=10, multiple:bool=False) -> List[int]:
    """
    Searches the reference dataset for most similar images to the
    query image

    Args:
        query_list: list containing path to all query images
        mask_list: list containing path to masks of all query images
        descriptor: method used to compute features
        metric: similarity measure to compare images
        bins: number of bins to use for histogram
        k: to return top k images that are most similar

    Returns:
        a list of lists indices of top k images in reference dataset
        that are most similar to query image
    """
    
    # define particular descriptors and distance metrics to be used
    # so that you don't have to input the function arguments every 
    # time you call the function
    extract_features_func = partial(extract_features, descriptor=descriptor,bins=bins)
    distance_func = partial(compute_distance, metric=metric)
    
    def _extract_features_from_path_unique(_path):
        return extract_features_func(path2img(_path))
    
    def _extract_features_from_path(_path):
        return [extract_features_func(path2img(_path)), ]

    def _extract_features_from_path_mask(_path,_path_mask):
        img, mask = path2img(_path), path2mask(_path_mask)
        if multiple:
            masks = extract_paintings_from_mask(mask)
            #print(f"{_path_mask} - {len(masks)}")
            return [extract_features_func(img, mask=m) 
                    for m in masks]
        return [extract_features_func(img, mask = mask), ]

    result_list_of_lists = []
    with mp.Pool(processes=20) as p:
        if mask_list is not None:
            query_descriptors = p.starmap(_extract_features_from_path_mask, [(query_path, mask_path) for query_path, mask_path in zip(query_list, mask_list)])
        else:
            query_descriptors = p.map(_extract_features_from_path, [query_path for query_path in query_list])
            
        image_descriptors = p.map(_extract_features_from_path_unique, [ref_path for ref_path in museum_list])
        for q in range(len(query_list)):
            query_list = []
            for i in range(len(query_descriptors[q])):
                query_feature = query_descriptors[q][i]
                dist = p.starmap(distance_func, [(query_feature, ref_feature) for ref_feature in image_descriptors])
                nearest_indices = np.argsort(dist)[:k]
                result_list = [index for index in nearest_indices]
                query_list.append(result_list)
            result_list_of_lists.append(query_list)
    return result_list_of_lists

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Content Based Image Retrieval')

    parser.add_argument(
        "--pickle","-p", action="store_true",
        help = "Generate pickle file with results")

    parser.add_argument(
        "--plot","-v", action="store_true",
        help = "show retrieval results for a random query image")

    parser.add_argument(
        "--use_masks", action="store_true",
        help = "whether to use masks for histogram generation or not. Using masks helps \
                us improve our features by extract the painting(foreground) from the \
                background and also removing any text present on the painting")

    parser.add_argument(
        "--multiple", action="store_true",
        help = "whether several paintings can appear in the mask or not (with a current maximum of 2).")

    parser.add_argument(
        "--museum_path","-r", default="/home/adityassrana/datatmp/Datasets/museum_dataset/processed/BBDD",
        help = "path to reference museum dataset. Example input: 'data/BBDD'")

    parser.add_argument(
        "--query_path","-q", default="/home/adityassrana/datatmp/Datasets/museum_dataset/processed/qsd1_w1",
        help = "path to query museum dataset. Example input: 'data/qsd1_w1'")

    parser.add_argument(
        "--descriptor","-d", default="rgb_histogram_1d",
        help = "descriptor for extracting features from image. DESCRIPTORS AVAILABLE: 1D and 3D Histograms - \
                gray_historam, rgb_histogram_1d, rgb_histogram_3d, hsv_histogram_1d, hsv_histogram_3d, lab_histogram_1d,\
                lab_histogram_3d, ycrcb_histogram_1d, ycrcb_histogram_3d. \
                Block and Pyramidal Histograms - lab_histogram_3d_pyramid and more.\
                lab_histogram_3d gives us the best results.")

    parser.add_argument(
        "--metric","-m", default="euclidean",
        help = "similarity measure to compare images. METRICS AVAILABLE: \
                cosine, manhattan, euclidean, intersect, kl_div, js_div bhattacharyya, hellinger, chisqr, correl. \
                hellinger and js_div give the best results.")

    parser.add_argument(
        "--bins","-b", default="64", type=int,
        help = "number of bins to use for histograms")

    parser.add_argument(
        "--map_k","-k", default="5", type=int,
        help = "Mean average precision of top-K results")

    args = parser.parse_args(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)

    # path to data
    museum_path = args.museum_path
    query_path = args.query_path
    
    # indexing the images and ground truth
    museum_list = get_image_path_list(museum_path)
    query_list = get_image_path_list(query_path)
    # get ground truth
    ground_truth= get_ground_truth(query_path)
    
    # sanity check
    print(f'number of images in reference dataset is {len(museum_list)}')
    print(f'number of images in query dataset is {len(query_list)}')
    
    k = args.map_k
    print('---------Now Searching Batches--------')

    mask_list = None
    if args.use_masks:
        # index the masks when required
        mask_list = get_image_path_list(query_path, extension='png')
        print(f'number of masks in query dataset is {len(mask_list)}')
    
    result_list_of_lists = search_batch(query_list = query_list, mask_list=mask_list, descriptor = args.descriptor, metric = args.metric, bins= args.bins, k=k, multiple=args.multiple)

    map_k = mapk(ground_truth, result_list_of_lists, k=k)
    print(f'map@{k} for the current run is {map_k}')

    # extra functions for pickling and visualizing results
    # use -p in cmdline args
    if args.pickle:
        print(f'writing this list to results.pkl \n {result_list_of_lists}')
        with open('result.pkl', 'wb') as f:
            pickle.dump(result_list_of_lists, f)

    # use -v in cmdline args
    if args.plot:
        query_index = np.random.randint(0,30)
        plot_results([query_list[query_index]] + [museum_list[index] for index in result_list_of_lists[query_index]])