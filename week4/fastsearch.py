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
from histograms import extract_features, extract_textures
from evaluation import mapk
from masks import extract_paintings_from_mask, generate_text_mask
from text_analysis import extract_text, compare_texts
from textboxes import generate_text_mask

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
        if query_params["text"] is not None:
            print("[WARNING] No text_list specified => text metric will not be used at all")
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
        
        all_results = []
        weights = []
        if query_params["color"] is not None:
            print('color being used')
            extract_features_func = partial(extract_features, descriptor=query_params["color"]["descriptor"],bins=query_params["color"]["bins"])
            color_distance_func = partial(compute_distance, metric=query_params["color"]["metric"])
            # descriptors extraction
            query_descriptors = p.map(lambda query: [extract_features_func(img, mask=m) for (img, m) in query], queries)
            image_descriptors = p.map(lambda path: extract_features_func(path2img(path)), museum_list)
            
            # comparison against database. Score is weighted with the value from params.
            results = [[p.starmap(lambda q, db: query_params["color"]["weight"] * color_distance_func(q, db), 
                                 [(query_desc, db_desc) for db_desc in image_descriptors])
                       for query_desc in query_descs]
                       for query_descs in query_descriptors]
            
            all_results.append(results)
            
        if query_params["texture"] is not None:
            print('texture being used')
            extract_features_func = partial(extract_textures, descriptor=query_params["texture"]["descriptor"],bins=query_params["texture"]["bins"])
            color_distance_func = partial(compute_distance, metric=query_params["texture"]["metric"])
            # descriptors extraction
            query_descriptors = p.map(lambda query: [extract_features_func(img, mask=m) for (img, m) in query], queries)
            image_descriptors = p.map(lambda path: extract_features_func(path2img(path)), museum_list)
            
            # comparison against database. Score is weighted with the value from params.
            results = [[p.starmap(lambda q, db: query_params["texture"]["weight"] * color_distance_func(q, db), 
                                 [(query_desc, db_desc) for db_desc in image_descriptors])
                       for query_desc in query_descs]
                       for query_descs in query_descriptors]
            
            all_results.append(results)
            
        if query_params["text"] is not None:
            print('text being used')
            text_distance_func = partial(compare_texts, similarity=query_params["text"]["metric"])
            # descriptors extraction
            query_descriptors = p.starmap(extract_txt, zip(queries, text_list))
            image_descriptors = p.map(read_GT_txt, museum_list)
            # comparison against database. Score is weighted with the value from params.
            results = [[p.starmap(lambda q, db: query_params["text"]["weight"] * (text_distance_func(q, db)), 
                                 [(query_desc, db_desc) for db_desc in image_descriptors])
                       for query_desc in query_descs]
                       for query_descs in query_descriptors]
            
            all_results.append(results)
        
        if len(all_results) == 0:
            print("[ERROR] You did not specify any feature extraction method.")
            return None
            
        # we sum the color/text/textures scores for each query and retrieve the best ones
        dist = np.sum(np.array(all_results), axis=0)
        for q in range(len(queries)):
            qlist = []
            for sq in range(len(queries[q])):
                dist = np.array(all_results[0][q][sq])
                for f in range(1, len(all_results)):
                    dist += all_results[f][q][sq]
                nearest_indices = np.argsort(dist)[:k]
                result_list = [index for index in nearest_indices]
                qlist.append(result_list)
            result_list_of_lists.append(qlist)
            
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

    parser.add_argument(
        "--use_color", action="store_true",
        help = "whether color matching will be used to do the retrieval.")

    parser.add_argument(
        "--use_text", action="store_true",
        help = "whether author matching will be used to do the retrieval.")
    
    parser.add_argument(
        "--use_texture", action="store_true",
        help = "whether texture matching will be used to do the retrieval.")
    
    # MASKS
    parser.add_argument(
        "--masks_multiple", action="store_true",
        help = "whether several paintings can appear in the mask or not (with a current maximum of 2).")

    # COLOR
    parser.add_argument(
        "--color_weight","-cow", type=float, default="0.33",
        help = "weight for the color matching")
    
    parser.add_argument(
        "--color_descriptor", default="lab_histogram_3d_blocks",
        help = "descriptor for extracting features from image. DESCRIPTORS AVAILABLE: 1D and 3D Histograms - \
                gray_historam, rgb_histogram_1d, rgb_histogram_3d, hsv_histogram_1d, hsv_histogram_3d, lab_histogram_1d,\
                lab_histogram_3d, ycrcb_histogram_1d, ycrcb_histogram_3d. \
                Block and Pyramidal Histograms - lab_histogram_3d_pyramid and more.\
                lab_histogram_3d_blocks gives us the best results.")

    parser.add_argument(
        "--color_metric", default="hellinger",
        help = "similarity measure to compare images. METRICS AVAILABLE: \
                cosine, manhattan, euclidean, intersect, kl_div, js_div bhattacharyya, hellinger, chisqr, correl. \
                hellinger and js_div give the best results.")

    parser.add_argument(
        "--color_bins", default="8", type=int,
        help = "number of bins to use for histograms")
    
    # TEXT
    parser.add_argument(
        "--text_weight","-txw", type=float, default="0.33",
        help = "weight for the text matching")
    
    parser.add_argument(
        "--text_reader", type=str, default="tesseract",
        help = "OCR algorithm used to extract the text from inside the textbox. READERS AVAILABLE: tesseract")
    
    parser.add_argument(
        "--text_metric", type=str, default="ratcliff_obershelp",
        help = "Metric used to compare extracted text with paintings text from the database. METRICS AVAILABLE: ratcliff_obershelp, levenshtein, cosine")
    
    # TEXTURES
    parser.add_argument(
        "--texture_weight","-tuw", type=float, default="0.33",
        help = "weight for the color matching")
    
    parser.add_argument(
        "--texture_descriptor", default="dct_blocks",
        help = "descriptor for extracting textures from image. DESCRIPTORS AVAILABLE: 1D and 3D Histograms - \
                gray_historam, rgb_histogram_1d, rgb_histogram_3d, hsv_histogram_1d, hsv_histogram_3d, lab_histogram_1d,\
                lab_histogram_3d, ycrcb_histogram_1d, ycrcb_histogram_3d. \
                Block and Pyramidal Histograms - lab_histogram_3d_pyramid and more.\
                lab_histogram_3d gives us the best results.")

    parser.add_argument(
        "--texture_metric", default="correl",
        help = "textures similarity measure to compare images. METRICS AVAILABLE: \
                cosine, manhattan, euclidean, intersect, kl_div, js_div bhattacharyya, hellinger, chisqr, correl. \
                hellinger and js_div give the best results.")

    parser.add_argument(
        "--texture_bins", default="8", type=int,
        help = "number of bins to use for textures histograms")

    
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
    if args.use_color:
        ordered_args["color"] = {
            "weight": args.color_weight,
            "descriptor": args.color_descriptor,
            "metric": args.color_metric,
            "bins": args.color_bins,
        }
    if args.use_text:
        ordered_args["text"] = {
            "weight": args.text_weight,
            "extractor": args.text_reader,
            "metric": args.text_metric,
        }
    if args.use_texture:
        ordered_args["texture"] = {
            "weight": args.texture_weight,
            "descriptor": args.texture_descriptor,
            "metric": args.texture_metric,
            "bins": args.texture_bins,
            
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
    if (args.filter_text or args.use_text) and os.path.exists(textboxes_path):
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