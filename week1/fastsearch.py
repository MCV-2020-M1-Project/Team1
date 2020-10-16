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

def get_image_path_list(data_path:str) -> List[Path]:
    """
    Used for indexing all images in a folder
    
    Args:
        data_path: path to folder containing the images
    
    Returns: 
        a list of paths to all the images in the folder
    """
    return sorted(glob.glob(os.path.join(data_path,'*.jpg')))

def get_image_mask_list(data_path:str) -> List[Path]:
    """
    Used for indexing all masks in a folder
    
    Args:
        data_path: path to folder containing the images
    
    Returns: 
        a list of paths to all the images in the folder
    """
    return sorted(glob.glob(os.path.join(data_path,'*.png')))

def path2img(path:Path) -> np.ndarray:
    """
    Convert image to numpy array from path
    
    Args: 
        path: path to image

    Returns: 
        a numpy array image from specified path
    """
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

def path2img_np(path:str) -> np.ndarray:
    """
    Convert image to numpy array from path using
    PIL and numpy. Image is loaded in RGB color
    space instead of BGR.
    
    Args: 
        path: path to image

    Returns: 
        a numpy array image from specified path
    """
    return np.array(Image.open(path))

def plot_results(image_path_list: List[Path]) -> None:
    """
    Show top k-images fetched from museum dataset
    that are most similar to the query images
    """
    fig,axes = plt.subplots(3,2, figsize=(16,8))
    for ax, path, title in zip(axes.flatten(), image_path_list ,['query image','retrieval1','retrieval2','retrieval3','retrieval4','retrieval5']):
        ax.imshow(path2img_np(path))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def get_ground_truth(data_path:str=None) -> List[List[int]]:
    """
    Retriev pickle file containing ground truths
    of query images
    """
    gt_file_list = glob.glob(os.path.join(data_path,'*gt*.pkl'))
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

def search_batch(query_list:List[Path]=None, descriptor:str='rgb_histogram_1d', metric:str="euclidean", bins:int=64, k:int=10) -> List[int]:
    """
    Searches the reference dataset for most similar images to the
    query image

    Args:
        query_list: list containing path to all query images
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
    #time you call the function
    extract_features_func = partial(extract_features, descriptor=descriptor,bins=bins)
    distance_func = partial(compute_distance, metric=metric)
    
    def _extract_features_from_path(_path):
        return extract_features_func(path2img(_path))

    result_list_of_lists = []
    with mp.Pool(processes=20) as p:
        
        query_descriptors = p.map(_extract_features_from_path, [query_path for query_path in query_list])
        image_descriptors = p.map(_extract_features_from_path, [ref_path for ref_path in museum_list])
    
        for q in range(len(query_list)):
            query_feature = query_descriptors[q]
            dist = p.starmap(distance_func, [(query_feature, ref_feature) for ref_feature in image_descriptors])
            nearest_indices = np.argsort(dist)[:k]
            result_list = [index for index in nearest_indices]
            result_list_of_lists.append(result_list)
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
        "--museum_path","-r", default="/home/adityassrana/datatmp/Datasets/museum_dataset/processed/BBDD",
        help = "path to reference museum dataset. Example input: 'data/BBDD'")

    parser.add_argument(
        "--query_path","-q", default="/home/adityassrana/datatmp/Datasets/museum_dataset/processed/qsd2_w1",
        help = "path to query museum dataset. Example input: 'data/qsd1_w1'")

    parser.add_argument(
        "--descriptor","-d", default="rgb_histogram_1d",
        help = "descriptor for extracting features from image. DESCRIPTORS AVAILABLE: \
                gray_historam, rgb_histogram_1d, rgb_histogram_3d, hsv_histogram_1d, hsv_histogram_3d, lab_histogram_1d,\
                lab_histogram_3d, ycrcb_histogram_1d, ycrcb_histogram_3d")

    parser.add_argument(
        "--metric","-m", default="euclidean",
        help = "similarity measure to compare images. METRICS AVAILABLE: \
                cosine, manhattan, euclidean, intersect, kl_div, bhattacharyya, hellinger, chisqr, correl")

    parser.add_argument(
        "--bins","-b", default="64", type=int,
        help = "number of bins to use for histograms")

    parser.add_argument(
        "--map_k","-k", default="5", type=int,
        help = "Mean average precision at top-K results")

    args = parser.parse_args(args)
    return args

def search_batch_mask(query_list:List[Path]=None, descriptor:str='rgb_histogram_1d', metric:str="euclidean", bins:int=64, k:int=10) -> List[int]:
    """
    Searches the reference dataset for most similar images to the
    query image

    Args:
        query_list: list containing path to all query images
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
    #time you call the function
    extract_features_func = partial(extract_features, descriptor=descriptor,bins=bins)

    distance_func = partial(compute_distance, metric=metric)
    
    def _extract_features_from_path(_path):
        return extract_features_func(path2img(_path))

    def _extract_features_from_path_mask(_path,_path_mask):
        return extract_features_func(path2img(_path), mask=path2mask(_path_mask))

    result_list_of_lists = []
    with mp.Pool(processes=20) as p:
        
        query_descriptors = p.starmap(_extract_features_from_path_mask, [(query_path,mask_path) for query_path, mask_path in zip(query_list, mask_list)])
        image_descriptors = p.map(_extract_features_from_path, [ref_path for ref_path in museum_list])
    
        for q in range(len(query_list)):
            query_feature = query_descriptors[q]
            dist = p.starmap(distance_func, [(query_feature, ref_feature) for ref_feature in image_descriptors])
            nearest_indices = np.argsort(dist)[:k]
            result_list = [index for index in nearest_indices]
            result_list_of_lists.append(result_list)
    return result_list_of_lists

if __name__ == '__main__':
    args = parse_args()
    print(args)

    #path to data
    museum_data = args.museum_path
    query_data = args.query_path
    print(query_data)
    
    #indexing the images and ground truth
    museum_list = get_image_path_list(museum_data)
    query_list = get_image_path_list(query_data)
    ground_truth= get_ground_truth(query_data)

    mask_list = get_image_mask_list(query_data)

    print(f'number of images in reference dataset is {len(museum_list)}')
    print(f'number of images in query dataset is {len(query_list)}')
    print(f'number of masks in query dataset is {len(query_list)}')

    #results = search(query_list[6],k=10)
    #print(f'Most similar images in the reference dataset are {results}')

    k = args.map_k
    print('---------Now Searching Batches--------')
    result_list_of_lists = search_batch_mask(query_list = query_list, descriptor = args.descriptor, metric = args.metric, bins= args.bins, k=k)
    print(result_list_of_lists)
    map_k = mapk(ground_truth, result_list_of_lists, k=k)
    print(f'map@{k} for the current run is {map_k}')

    if args.pickle:
        print(f'writing this list to results.pkl \n {result_list_of_lists}')
        with open('result.pkl', 'wb') as f:
            pickle.dump(result_list_of_lists, f)

    if args.plot:
        query_index = 2
        plot_results([query_list[query_index]] + [museum_list[index] for index in result_list_of_lists[query_index]])