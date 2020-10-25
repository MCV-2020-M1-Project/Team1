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


def __opening(m, size=(45, 45)):
    kernel = np.ones(size, np.uint8) 
    m = cv2.erode(m, kernel, iterations=1) 
    m = cv2.dilate(m, kernel, iterations=1) 
    return m

def __closing(m, size=(45, 45)):
    kernel = np.ones(size, np.uint8) 
    m = cv2.dilate(m, kernel, iterations=1) 
    m = cv2.erode(m, kernel, iterations=1) 
    return m
    
def __function_4vPolygon_F1(m_orig, p1,p2,p3,p4):
    m = np.zeros(m_orig.shape[:2])
    pts = np.array([p1,p2,p3,p4], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillConvexPoly(m, pts, True)
    tp = int(np.sum(m * m_orig))
    fn = int(np.sum((1 - m) * m_orig))
    try:
        return evaluate_mask(m, m_orig)[2]#compute_iou(m, MASK)
    except:
        return 0

def opening_mask(m):
    # relative
    val = min(m.shape)
    size = (min(int(0.1 * val), 100), min(int(0.1*val), 100))
    return __opening(m, size=size)

def closing_mask(m):
    # relative
    val = min(m.shape)
    size = (min(int(0.1 * val), 100), min(int(0.1*val), 100))
    return __closing(m, size=size)

def improve_mask_v0(m_orig):
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
    # we do it per painting inside the mask (connected component)
    for f in filters:
        final_mask = np.zeros_like(mask)
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        for lab in range(1, num_labels):      
            m = (labels == lab).astype(np.uint8)
            final_mask += f(m).astype(np.uint8)
        mask = final_mask
    return final_mask