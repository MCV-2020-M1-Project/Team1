from histograms import gray_historam
from masks import generate_text_mask, extract_paintings_from_mask, extract_biggest_connected_component
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import pickle as pkl
from utils import opening, closing
import argparse
import time
from tqdm import tqdm
from utils import closing
from evaluation import compute_iou

def evaluate_textboxes(gt_boxes, boxes):
    """
    Evaluates the mean intersection-over-union between GT textboxes and the given ones.
    
    Parameters
    ----------
    gt_boxes : list of textboxes for each image, as described in W2 slides
    boxes : list of textboxes for each image, as described in W2 slides
    
    Returns
    -------
    mean_iou: mean intersection-over-union
    """
    assert len(gt_boxes) == len(boxes)
    
    iou = 0
    # compute IOU per image
    for i in range(len(boxes)):
        if len(boxes[i]) == 0 or len(gt_boxes[i]) == 0:
            continue
            
        max_dim = np.max(np.max(boxes[i]))
        shape = (max_dim, max_dim)
        # We compute the IOU by generating both masks with all given textboxes highlighted.
        gt_mask, mask = generate_text_mask(shape, gt_boxes[i]), generate_text_mask(shape, boxes[i])
        iou += compute_iou(gt_mask, mask)
    return iou / len(boxes)

def brightText(img):
    """
    Generates the textboxes candidated based on TOPHAT morphological filter.
    Works well with bright text over dark background.
    
    Parameters
    ----------
    img : ndimage to process
    
    Returns
    -------
    mask: uint8 mask with regions of interest (possible textbox candidates)
    """
    kernel = np.ones((30, 30), np.uint8) 
    img_orig = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    TH = 150
    img_orig[(img_orig[:,:,0] < TH) | (img_orig[:,:,1] < TH) | (img_orig[:,:,2] < TH)] = (0,0,0)
    
    img_orig = closing(img_orig, size=(1, int(img.shape[1] / 8)))
    
    return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)
        

def darkText(img):
    """
    Generates the textboxes candidated based on BLACKHAT morphological filter.
    Works well with dark text over bright background.
    
    Parameters
    ----------
    img : ndimage to process
    
    Returns
    -------
    mask: uint8 mask with regions of interest (possible textbox candidates)
    """
    kernel = np.ones((30, 30), np.uint8) 
    img_orig = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    TH = 150
    img_orig[(img_orig[:,:,0] < TH) | (img_orig[:,:,1] < TH) | (img_orig[:,:,2] < TH)] = (0,0,0)
    
    img_orig = closing(img_orig, size=(1, int(img.shape[1] / 8)))
    
    return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)


def get_textbox_score(m, p_shape):
    """
    Generates a score for how textbox-ish a mask connected component is.
    
    Parameters
    ----------
    m : mask with the textbox region with 1's
    p_shape : shape of the minimum bounding box enclosing the painting.
    
    Returns
    -------
    score: score based on size + shape
    """
    m = m.copy()
    
    # we generate the minimum bounding box for the extracted mask
    x,y,w,h = cv2.boundingRect(m.astype(np.uint8))
    
    # some upper and lower thresholding depending on its size and the painting size.
    if w < 10 or h < 10 or h > w:
        return 0
    if w >= p_shape[0]*0.8 or h >= p_shape[1]/4:
        return 0

    # we compute the score according to its shape and its size
    sc_shape = np.sum(m[y:y+h, x:x+w]) / (w*h)
    sc_size = (w*h) / (m.shape[0] * m.shape[1])
    
    final_score = (sc_shape + 50*sc_size) / 2
        
    return final_score

def get_best_textbox_candidate(mask, original_mask):
    """
    Analyzes all connected components and returns the best one according to the textbox metric.
    
    Parameters
    ----------
    m : mask with the textboxes regions with 1's
    original_mask : painting mask (size of the whole image)
    
    Returns
    -------
    score: score based on size + shape
    """
    # we will need it to crop the final textbox region so it does not goes beyond painting limits.
    x,y,w,h = cv2.boundingRect(original_mask.astype(np.uint8))
    p_shape = (w,h)
    p_coords = (x,y)
    
    # we get the biggest connected component with a score higher than TH as the textbox proposal
    mask_c = mask.copy()
    TH = 0.5
    i = 0
    found = False
    mask = None
    best_sc = 0
    while not found:
        biggest = extract_biggest_connected_component(mask_c).astype(np.uint8)
        if np.sum(biggest) == 0:
            return 0, None
        
        sc = get_textbox_score(biggest, p_shape)
        
        if sc > TH:
            mask = biggest
            best_sc = sc
            found = True
        else:
            mask_c -= biggest
            
    # we crop it and give it a margin dependant on the painting size.
    x, y, w, h = cv2.boundingRect(mask)
    M_W = 0.05
    M_H = 0.05
    ref = min(p_shape)
    x0,y0,x,y = (x - int(ref*M_W/2), y - int(ref*M_H/2), 
            (x+w) + int(ref*M_W/2), (y+h) + int(ref*M_H/2))
    return best_sc, [max(0,x0), max(0,y0), min(x, p_coords[0] + p_shape[0]), min(y, p_coords[1] + p_shape[1])]

def extract_textbox(orig_img, mask=None):
    """
    Given an image (and a mask), extracts the textbox in it, if any found (taking into account a metric).
    
    Parameters
    ----------
    orig_img : image from which extract the textbox
    mask : mask to use. In case of multiple paintings, each connected component will be considered as a different
            painting.
    
    Returns
    -------
    score: bboxes (0 or 1 per connected component found in the mask)
    """
    masks = []
    shapes = []
    bboxes = []
    # extract each painting's mask
    if mask is None:
        masks = [np.ones(orig_img.shape[:2])]
    else:
        masks = extract_paintings_from_mask(mask)
        
    if masks is None:
        return [[0,0,0,0]]
    # we try to extract one textbox per painting
    for m in masks:
        img = orig_img.copy()
        img[m == 0] = (0,0,0)

        sc_br, bbox_br = get_best_textbox_candidate(brightText(img), m)
        sc_dr, bbox_dr = get_best_textbox_candidate(darkText(img), m)
        bbox = bbox_br
        if sc_dr == 0 and sc_br == 0:
            continue
        if sc_dr > sc_br:
            bbox = bbox_dr
        bboxes.append(bbox)

    return bboxes


def generate_text_mask(shape, textboxes):
    """
    Given a shape and textboxes list, it returns a mask with the textboxes with values at 1's.
    
    Parameters
    ----------
    shape : resulting mask shape. Has to be big enough to enclose the textboxes or will crash.
    textboxes : list of textboxes to show, with the format specified in W2 slides.
    
    Returns
    -------
    mask: mask with textboxes at 1
    """
    if textboxes is None or len(textboxes) == 0:
        return np.zeros(shape).astype(np.uint8)
    
    mask = np.zeros(shape)
    for (xtl, ytl, xbr, ybr) in textboxes:
        pts = np.array(((xtl, ytl), (xtl, ybr), (xbr, ybr), (xbr, ytl)))
        cv2.fillConvexPoly(mask, pts, True)
    return mask.astype(np.uint8)


TEXTBOX_RETRIEVERS = {
    "v0": extract_textbox
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates textboxes from images (using masks if available). Output file will be stored in the execution path if unspecified.")
    parser.add_argument('--query', help="Path to query folder.", type=str)
    #parser.add_argument('--query', help="Path to query folder.", type=str)
    parser.add_argument('--use_masks', action="store_true", help="If masks are to be used.")
    #parser.add_argument('--masks', help="Path where masks are located. If empty, masks will be retrieved from query folder.", type=str)
    parser.add_argument('--retriever', help="Textbox retriever to use.", type=str, choices=list(TEXTBOX_RETRIEVERS.keys()),)
    #parser.add_argument('--output', help="Folder where output file 'text_boxes_{VERSION}.pkl' file will be stored.", type=str)
    args = parser.parse_args()


    if not os.path.exists(args.query):
        print(f"[ERROR] Query folder '{args.query}' does not exist.")
        exit()
        
    masks_folder = args.query
    if masks_folder and not os.path.exists(masks_folder):
        print(f"[ERROR] Masks folder '{masks_folder}' does not exist.")
        exit()
    elif args.use_masks:
        print("[WARNING] Masks path unspecified... The masks inside query folder will be used...")
        masks_folder = args.query
        
    output_folder = args.query
    if output_folder  and not os.path.exists(output_folder):
        print(f"Creating output folder '{output_folder}'...")
        os.mkdir(output_folder)


    print(f"Evaluating with textbox retriever >{args.retriever}<...")
    
    t0 = time.time()
    # We load the dataset metadata
    try:
        with open(os.path.join(args.query, "frames.pkl"), 'rb') as file:
            frames = pkl.load(file)
    except:
        print(f"[ERROR] Missing 'frames.pkl' file in folder '{args.query}'.")
        exit()
    db_count = len(frames)
    print(f"There are {db_count} images to process.")

    # We load the images and their associated masks
    images = [cv2.imread(os.path.join(args.query, f"{i:05d}.jpg"), 1) for i in range(db_count)]
    if args.use_masks:
        masks = [cv2.imread(os.path.join(masks_folder, f"{i:05d}.png"), 0) for i in range(db_count)]
    else:
        masks = [None for i in range(db_count)]
        
    bboxes = []
    for i in tqdm(range(len(masks))):
        bboxes.append(TEXTBOX_RETRIEVERS[args.retriever](images[i], mask=masks[i]))
        
    path = os.path.join(output_folder, f"textboxes_{args.retriever}.pkl")
    with open(path, 'wb+') as file:
        pkl.dump(bboxes, file)
    
    print(f"Done! Results saved to output file: '{path}'.")
    print(f"Successfully processed {len(masks)} images in {time.time() - t0:.2f}s!")


