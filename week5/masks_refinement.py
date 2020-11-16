import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import pickle as pkl
import multiprocessing.dummy as mp
from tqdm import tqdm
from masks import extract_mask, extract_paintings_from_mask
from evaluation import evaluate_mask, compute_iou
from utils import opening, closing
import argparse
import time
from tqdm import tqdm
import math
from angles import extract_angle, rotate_image, compute_angle, rotate_point
    
def __function_4vPolygon_F1(m_orig, p1,p2,p3,p4):
    # Function that computer the F1 score between the polygon formed by (p1, p2, p3, p4) and m_orig mask.
    m = np.zeros(m_orig.shape[:2])
    pts = np.array([p1,p2,p3,p4], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillConvexPoly(m, pts, True)
    try:
        return evaluate_mask(m, m_orig)[2]#compute_iou(m, MASK)
    except:
        return 0

def opening_mask(m, img=None):
    # we give some margin
    marg = 50
    aux = np.zeros((m.shape[0] + marg*2, m.shape[1] + marg*2))
    aux[marg:marg+m.shape[0], marg:marg+m.shape[1]] = m
    
    # relative to the shape of the mask to compute opening
    val = min(m.shape)
    size = (min(int(0.1 * val), 100), min(int(0.1*val), 100))
    aux = opening(aux, size=size)
    return aux[marg:marg+m.shape[0], marg:marg+m.shape[1]], None

def closing_mask(m, img=None):
    # we give some margin
    marg = 200
    aux = np.zeros((m.shape[0] + marg*2, m.shape[1] + marg*2))
    aux[marg:marg+m.shape[0], marg:marg+m.shape[1]] = m
    
    # relative to the shape of the mask to compute closing
    val = min(m.shape)
    size = (min(int(0.1 * val), 100), min(int(0.1*val), 100))
    aux =  closing(aux, size=size)
    return aux[marg:marg+m.shape[0], marg:marg+m.shape[1]], None

def improve_mask_v0(m_orig, img=None):
    # Algorithm explained in W2 slides. Gets the 4-vertices polygon with the maximum F1 score considering
    # the mask. It is a brute-force algorithm (can take up to 5 minutes in high-resolution images.
    x,y,w,h = cv2.boundingRect(m_orig.astype(np.uint8))
    pts = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
    best_pts = pts
    best_score = __function_4vPolygon_F1(m_orig, pts[0],pts[1],pts[2],pts[3])
    for k in range(len(pts)):
        for i in range(-15, 15, 3):
            for j in range(-15, 15, 3):
                pts2 = [pt.copy() for pt in pts]
                pts2[k][0] += i
                pts2[k][1] += j
                sc = __function_4vPolygon_F1(m_orig, pts2[0],pts2[1],pts2[2],pts2[3])
                if sc > best_score:
                    pts = pts2
                    best_pts = pts2
                    best_score = sc
    
    m_final = np.zeros(m_orig.shape[:2])
    cv2.fillConvexPoly(m_final, np.array(best_pts).reshape((-1,1,2)), True)
    return m_final*255, None

def improve_mask_angle(m_orig, img=None):
    if img is None:
        raise Exception("[ERROR] Image needed to use this filter: 'improve_mask_angle'.")
        
    angle = extract_angle(img)
    rot_angle = angle
    if rot_angle > 90:
        rot_angle = rot_angle - 180.0
    m_orig = rotate_image(m_orig, -rot_angle, mask=True) # we fix it
        
    # Algorithm explained in W2 slides. Gets the 4-vertices polygon with the maximum F1 score considering
    # the mask. It is a brute-force algorithm (can take up to 5 minutes in high-resolution images.
    x,y,w,h = cv2.boundingRect(m_orig.astype(np.uint8))
    # top-left, top-right, bottom-right, bottom-left
    pts = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
    best_pts = pts
    best_score = __function_4vPolygon_F1(m_orig, pts[0],pts[1],pts[2],pts[3])
    
    to_try_i = list(range(0, 15, 3)) + list(range(-15, 0, 3))
    to_try_j = list(range(0, 15, 3)) + list(range(-15, 0, 3))
    for k in range(len(pts)):
        for i in to_try_i:
            for j in to_try_j:
                pts2 = [pt.copy() for pt in pts]
                pts2[k][0] += i
                pts2[k][1] += j
                
                # EXTRA STEP COMPARED TO v0: check angle of deviation and if 
                # norm of the vector with both angle deviations > 3, we continue
                k_p = k-1 if k>0 else len(pts)-1
                k_n = k+1 if k<len(pts)-1 else 0
                v0 = (pts2[k][0] - pts2[k_p][0], pts2[k][1] - pts2[k_p][1])
                v1 = (pts2[k][0] - pts2[k_n][0], pts2[k][1] - pts2[k_n][1])
                angle_p = compute_angle(v0, (1,0))
                angle_n = compute_angle(v1, (1,0))
                array = (min(angle_p % 90, (90.0-angle_p) % 90), min(angle_n % 90, (90.0-angle_n) % 90))
                norm = np.linalg.norm(array)
                if norm > 3:
                    continue
                
                # evaluate this case
                sc = __function_4vPolygon_F1(m_orig, pts2[0],pts2[1],pts2[2],pts2[3])
                if sc > 1.05 * best_score:
                    pts = pts2
                    best_pts = pts2
                    best_score = sc
    
    m_final = np.zeros(m_orig.shape[:2])
    cv2.fillConvexPoly(m_final, np.array(best_pts).reshape((-1,1,2)), True)
    
    metadata = [angle, [rotate_point(img, pt, rot_angle) for pt in best_pts]]
    return rotate_image(m_final*255, rot_angle, mask=True), metadata # we rotate it back
    

def apply_filters(mask, filters, img=None):
    mask_copy = mask.copy()
    metadata_list = []
    # apply consecutive filters to a mask.
    for i in range(len(filters)):
        f = filters[i]
        final_mask = np.zeros_like(mask)
        masks = extract_paintings_from_mask(mask)
        #num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        #for lab in range(1, num_labels):     
            #m = (labels == lab).astype(np.uint8) 
            
        # we will always have a painting
        if masks is None or len(masks) == 0:
            x, y, w, h = cv2.boundingRect(mask_copy)
            mask_copy[y:y+h, x:x+w] = 255
            return mask_copy, (extract_angle(img) if img is not None else 0.0, [[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
        
        for m in masks:
            _,_, w, h = cv2.boundingRect(m)
            if w / mask.shape[1] < 0.15 or h / mask.shape[1] < 0.15:
                continue
            f_m, metadata = f(m, img=img)
            final_mask += f_m.astype(np.uint8)
            # last filter gives us the metadata: [angle, [p1,p2,p3,p4]]
            if i == len(filters) - 1:
                metadata_list.append(metadata)
        mask = final_mask
        
        """
    # we will always have a painting
    x, y, w, h = cv2.boundingRect(final_mask)
    if w / mask.shape[1] < 0.10 or h / mask.shape[1] < 0.10:
        x, y, w, h = cv2.boundingRect(mask_copy)
        mask_copy[y:y+h, x:x+w] = 255
        return mask_copy, (extract_angle(img) if img is not None else 0.0, [[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
        """
    return final_mask, metadata_list

def refine(mask, refiner="v0", img = None):
    return apply_filters(mask, MASK_REFINERS[refiner]["filters"], img=img)

MASK_REFINERS = {
    "v0": {
            "filters": (closing_mask, opening_mask, improve_mask_v0),
    },
    "angle": {
            "filters": (closing_mask, opening_mask, improve_mask_angle),
    },
    "simple": {
            "filters": (closing_mask, opening_mask),
    },
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Refines the previously generated masks (from Task of Week 1). IMPORTANT: a 'frames.pkl' file is needed in the masks folder.")
    parser.add_argument('--masks', help="Path to masks to refine.", type=str)
    parser.add_argument('--refiner', help="Mask refiner method to use.", type=str, choices=list(MASK_REFINERS.keys()),)
    parser.add_argument('--output', help="Path to folder where refined masks will be stored.", type=str,)
    args = parser.parse_args()


    if not os.path.exists(args.masks):
        print(f"[ERROR] Masks folder '{args.masks}' does not exist.")
        exit()
    if not os.path.exists(args.output):
        print(f"Creating output folder '{args.output}'...")
        os.mkdir(args.output)


    print(f"Evaluating with mask refiner >{args.refiner}<...")
    
    t0 = time.time()
    # We load the dataset metadata
    print(args.masks)
    try:
        with open(os.path.join(args.masks, "frames.pkl"), 'rb') as file:
            frames = pkl.load(file)
    except:
        print(f"[ERROR] Missing 'frames.pkl' file in folder '{args.masks}'.")
        exit()
    db_count = len(frames)
    print(f"There are {db_count} masks to refine.")

    # We load the images and their associated masks
    masks = [cv2.imread(os.path.join(args.masks, f"{i:05d}.png"), 0) for i in range(db_count)]
    images = [cv2.imread(os.path.join(args.masks, f"{i:05d}.jpg"), 1) for i in range(db_count)]
    processed_masks = []
    metadata_list = []
    generate_pickle = True
    for i in tqdm(range(len(masks))):
        path = os.path.join(args.output, f"{i:05d}.png")
        if os.path.exists(path):
            print(f"[WARNING] Pickle file will not be generated.")
            generate_pickle = False
            print(f"[{i}] Already generated. Skipping...")
            continue
            
        m, metadata = apply_filters(masks[i], MASK_REFINERS[args.refiner]["filters"], img=images[i])
        processed_masks.append(m)
        metadata_list.append(metadata)
        
        # we save the generated masks
        cv2.imwrite(path, m)
    
    if generate_pickle:
        print(f'writing this list to results.pkl \n {metadata_list}')
        with open(os.path.join(args.output, 'frames.pkl'), 'wb') as f:
            pkl.dump(metadata_list, f)
        
    print(f"Successfully refined {len(masks)} masks in {time.time() - t0:.2f}s!")
