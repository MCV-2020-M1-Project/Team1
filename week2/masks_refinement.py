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
from masks import extract_mask
from evaluation import evaluate_mask, compute_iou
from utils import opening, closing
import argparse
import time
from tqdm import tqdm
    
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

def opening_mask(m):
    # relative to the shape of the mask to compute opening
    val = min(m.shape)
    size = (min(int(0.1 * val), 100), min(int(0.1*val), 100))
    return opening(m, size=size)

def closing_mask(m):
    # relative to the shape of the mask to compute closing
    val = min(m.shape)
    size = (min(int(0.1 * val), 100), min(int(0.1*val), 100))
    return closing(m, size=size)

def improve_mask_v0(m_orig):
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
    return m_final*255
    

def apply_filters(mask, filters):
    # apply consecutive filters to a mask.
    for f in filters:
        final_mask = np.zeros_like(mask)
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        for lab in range(1, num_labels):      
            m = (labels == lab).astype(np.uint8)
            final_mask += f(m).astype(np.uint8)
        mask = final_mask
    return final_mask



MASK_REFINERS = {
    "v0": {
            "filters": (closing_mask, opening_mask, improve_mask_v0),
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
    processed_masks = []
    for i in tqdm(range(len(masks))):
        path = os.path.join(args.output, f"{i:05d}.png")
        if os.path.exists(path):
            print(f"[{i}] Already generated. Skipping...")
            continue
            
        m = apply_filters(masks[i], MASK_REFINERS[args.refiner]["filters"])
        processed_masks.append(m)
        
        # we save the generated masks
        cv2.imwrite(path, m)
    
    print(f"Successfully refined {len(masks)} masks in {time.time() - t0:.2f}s!")
