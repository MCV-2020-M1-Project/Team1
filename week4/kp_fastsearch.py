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
import time

from distances import compute_distance
from histograms import extract_features, extract_textures
from evaluation import mapk
from masks import extract_paintings_from_mask, generate_text_mask
from text_analysis import extract_text, compare_texts
from textboxes import generate_text_mask
from keypoints import extract_keypoints
from local_descriptors import extract_local_descriptors
from kp_matching import match_keypoints_descriptors

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



def read_GT_txt(img_path):
    txt_path = img_path.replace('jpg', 'txt')
    if not os.path.exists(txt_path):
        print(f"ERROR: path '{txt_path}' does not exist.")

    with open(txt_path, "r") as f:
        txt = f.readline()
        split = txt.split("'")
        if len(split) > 2:
            return split[1]
    return None

def extract_txt(queries, textboxes):
    #print(textboxes)
    #print(f"{len(textboxes)} VS {len(queries)}")
    texts = []
    if len(queries) == len(textboxes):
        for (img, mask), txt in zip(queries, textboxes):
            min_x, min_y, max_x, max_y = txt
            extracted = extract_text(img[min_y:max_y, min_x:max_x], query_params["text"]["extractor"])
            texts.append(extracted if extracted != "" else None)
    else:
        for img, mask in queries:
            found = False
            i = 0
            while not found and i < len(textboxes):
                tb = textboxes[i]
                found = mask[tb[1]-1, tb[0]] != 0
                if not found:
                    i+=1

            if not found:
                texts.append(None)
            else:
                min_x, min_y, max_x, max_y = textboxes[i]
                txt = img[min_y:max_y, min_x:max_x]
                extracted = extract_text(txt, query_params["text"]["extractor"])
                texts.append(extracted if extracted != "" else None)

    return texts

# we need img, mask, text
def extract_color_metric(db_img, query_img, query_mask, query_txtboxes, params):
    
    
    query_features = extract_features(image=query_img, descriptor=params["descriptor"], bins=params["bins"], mask=query_mask)
    db_features = extract_features(image=query_img, descriptor=params["descriptor"], bins=params["bins"])
    
    return compute_distance(query_features, db_features, metric=params["metric"])

def _extract_image_and_mask_from_path(img_path, mask_path, textboxes, multiple):
    img = path2img(img_path)
    if mask_path is None:
        text_mask = 1 - generate_text_mask(img.shape[:2], textboxes) # returns all 1's if textboxes == None
        return [(img, text_mask), ]

    mask = path2mask(mask_path)
    text_mask = 1 - generate_text_mask(mask.shape, textboxes)
    if multiple:
        masks = extract_paintings_from_mask(mask)
        if masks is None or len(masks) == 0:
            return [(img, np.ones_like(img).astype(np.uint8)), ]
        return [(img, ((text_mask != 0) & (m != 0)).astype(np.uint8)) for m in masks]
    return [(img, ((text_mask != 0) & (mask != 0)).astype(np.uint8)), ]


def search_batch(museum_list:List[Path], query_list:List[Path], mask_list:List[Path], text_list, query_params, k=10) -> List[int]:
    """
    Searches the reference dataset for most similar images to the
    query image

    Args:
        query_list: list containing path to all query images
        mask_list: list containing path to masks of all query images
        text_list: list containing bboxes for each image
        query_params: dictionary containing info for all the features extractors used

    Returns:
        a list of lists indices of top k images in reference dataset
        that are most similar to query image
    """
    if text_list is None:
        print("[WARNING] No text_list specified => text will not be used at all")
        text_list = [None for l in query_list]
    
    if mask_list is None:
        if query_params["masks"] is not None:
            print("[WARNING] No mask_list specified => masks will not be used at all")
        mask_list = [None for l in query_list]
    
    result_list_of_lists = []
    with mp.Pool(processes=20) as p:
        queries = p.starmap(_extract_image_and_mask_from_path, 
                            [(query_list[q], 
                              mask_list[q], 
                              text_list[q], 
                              query_params["masks"] and query_params["masks"]["multiple"]) 
                             for q in range(len(query_list))])
        

        extract_kp_func = partial(extract_keypoints, method=query_params["keypoints"]["extractor"])
        extract_desc_func = partial(extract_local_descriptors, method=query_params["keypoints"]["descriptor"])

        # descriptors extraction
        t0 = time.time()
        query_descriptors = p.map(lambda query: [extract_desc_func(img, extract_kp_func(img)) for (img, m) in query], queries)
        print(f"[INFO] Query descriptors extracted: {time.time()-t0}s.")

        def desc_gallery(path):
            img = path2img(path)
            return extract_desc_func(img, extract_kp_func(img))
        t0 = time.time()
        image_descriptors = p.map(lambda path: desc_gallery(path), museum_list)
        print(f"[INFO] Gallery descriptors extracted: {time.time()-t0}s.")
        
        t0 = time.time()
        for q in range(len(query_list)):
            qlist = []
            for i in range(len(query_descriptors[q])):
                query_feature = query_descriptors[q][i]
                dists = p.starmap(match_keypoints_descriptors, 
                        [(query_feature, ref_feature) 
                        for ref_feature in image_descriptors])
                #dists = [match_keypoints_descriptors(query_feature.astype(np.float32), ref_feature.astype(np.float32)) 
                #        if ref_feature is not None else 0 for ref_feature in image_descriptors]
                nearest_indices = np.argsort(dists)[::-1][:k]
                print(np.sort(dists)[::-1][:k])
                if dists[nearest_indices[0]] == 0:
                    # no query match => -1
                    qlist.append([-1,])
                result_list = [index for index in nearest_indices]
                qlist.append(result_list)
            result_list_of_lists.append(qlist)
            
        print(f"[INFO] Results computed: {time.time()-t0}s.")
    return result_list_of_lists


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Content Based Image Retrieval')
    
    parser.add_argument(
        "--museum_path","-r", default="/home/adityassrana/datatmp/Datasets/museum_dataset/processed/BBDD",
        help = "path to reference museum dataset. Example input: 'data/BBDD'")

    parser.add_argument(
        "--query_path","-q", default="/home/adityassrana/datatmp/Datasets/museum_dataset/processed/qsd1_w1",
        help = "path to query museum dataset. Example input: 'data/qsd1_w1'")
    
    parser.add_argument(
        "--pickle","-p", action="store_true",
        help = "Generate pickle file with results")

    parser.add_argument(
        "--plot","-v", action="store_true",
        help = "show retrieval results for a random query image")

    # MODES THAT CAN BE USED
    parser.add_argument(
        "--use_masks", action="store_true",
        help = "whether to use masks for histogram generation or not. Using masks helps \
                us improve our features by extract the painting(foreground) from the \
                background and also removing any text present on the painting")

    parser.add_argument(
        "--filter_text", action="store_true",
        help = "whether textboxes will be loaded from 'text_boxes.pkl' file and removed from the color and texture feature extractions.")
    
    
    # MASKS
    parser.add_argument(
        "--masks_multiple", action="store_true",
        help = "whether several paintings can appear in the mask or not (with a current maximum of 2).")


    # KEYPOINTS
    parser.add_argument(
        "--extractor","-txw", type=str, default="surf",
        help = "method used to extract keypoints. AVAILABLE={sift, surf, orb, harris_corner_detector, harris_corner_subpixel, hl, dog, log, dog}")
    
    parser.add_argument(
        "--descriptor", type=str, default="surf",
        help = "descriptor used to extract keypoint features. AVAILABLE={sift, surf, root_sift, orb, daisy, hog, lbp}")
    
    parser.add_argument(
        "--matching", type=str, default="bruteforce",
        help = "Algorithm used to match keypoints. AVAILABLE={bruteforce, flann}")
    

    # EVALUATION
    parser.add_argument(
        "--map_k","-k", default="5", type=int,
        help = "Mean average precision of top-K results")

    args = parser.parse_args(args)
    return args


def from_args_to_query_params(args):
    ordered_args = {
        "masks": None, "color": None, "text": None, "texture": None,
        "filter_text": args.filter_text
    }
    if args.masks_multiple:
        ordered_args["masks"] = {
            "multiple": args.masks_multiple
        }
    ordered_args["keypoints"] = {
        "extractor": args.extractor,
        "descriptor": args.descriptor,
        "matching": args.matching,
    }
    return ordered_args
        

if __name__ == '__main__':
    args = parse_args()
    query_params = from_args_to_query_params(args)
    print(query_params)

    # path to data
    museum_path = args.museum_path
    query_path = args.query_path
    
    # indexing the images and ground truth
    museum_list = get_image_path_list(museum_path)
    query_list = get_image_path_list(query_path)
    
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
        assert len(mask_list) == len(query_list)
       
    text_list = None
    textboxes_path = os.path.join(query_path, "text_boxes.pkl")
    if args.filter_text and os.path.exists(textboxes_path):
        print("Textboxes loaded.")
        with open(textboxes_path, 'rb') as file:
            text_list = pickle.load(file)
    
    result_list_of_lists = search_batch(museum_list, 
                                        query_list, 
                                        mask_list, 
                                        text_list, 
                                        query_params,
                                        k = k)


        
    # extra functions for pickling and visualizing results
    # use -p in cmdline args
    if args.pickle:
        print(f'writing this list to results.pkl \n {result_list_of_lists}')
        with open(os.path.join(args.query_path, 'result.pkl'), 'wb') as f:
            pickle.dump(result_list_of_lists, f)
    
    # get ground truth
    ground_truth= get_ground_truth(query_path)
    map_k = mapk(ground_truth, result_list_of_lists, k=k)
    print(f'map@{k} for the current run is {map_k}')
    if k > 1:
        map_1 = mapk(ground_truth, result_list_of_lists, k=1)
        print(f'map@{1} for the current run is {map_1}')

    # use -v in cmdline args
    if args.plot:
        query_index = np.random.randint(0,30)
        plot_results([query_list[query_index]] + [museum_list[index] for index in result_list_of_lists[query_index]])