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

def evaluate_textboxes(gt_boxes, boxes):
    assert len(gt_boxes) == len(boxes)
    iou = 0
    for i in range(len(boxes)):
        if len(boxes[i]) == 0 or len(gt_boxes[i]) == 0:
            continue
        max_dim = np.max(np.max(boxes[i]))
        shape = (max_dim, max_dim)
        gt_mask, mask = generate_text_mask(shape, gt_boxes[i]), generate_text_mask(shape, boxes[i])
        iou += compute_iou(gt_mask, mask)
    return iou / len(boxes)

def compute_opening(m, size=(45, 45)):
    kernel = np.ones(size, np.uint8) 
    m = cv2.erode(m, kernel, iterations=1) 
    m = cv2.dilate(m, kernel, iterations=1) 
    return m

def compute_closing(m, size=(45, 45)):
    kernel = np.ones(size, np.uint8) 
    m = cv2.dilate(m, kernel, iterations=1) 
    m = cv2.erode(m, kernel, iterations=1) 
    return m

def brightText(img):
    kernel = np.ones((30, 30), np.uint8) 
    img_orig = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    TH = 150
    img_orig[(img_orig[:,:,0] < TH) | (img_orig[:,:,1] < TH) | (img_orig[:,:,2] < TH)] = (0,0,0)
    kernel = np.ones((1, int(img.shape[1] / 8)), np.uint8) 
    img_orig = cv2.dilate(img_orig, kernel, iterations=1) 
    img_orig = cv2.erode(img_orig, kernel, iterations=1) 
    
    return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)
        

def darkText(img):
    kernel = np.ones((30, 30), np.uint8) 
    img_orig = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    TH = 150
    img_orig[(img_orig[:,:,0] < TH) | (img_orig[:,:,1] < TH) | (img_orig[:,:,2] < TH)] = (0,0,0)
    kernel = np.ones((1, int(img.shape[1] / 8)), np.uint8) 
    img_orig = cv2.dilate(img_orig, kernel, iterations=1) 
    img_orig = cv2.erode(img_orig, kernel, iterations=1) 
    
    return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)


def get_textbox_score(m, p_shape):
    m = m.copy()
    #plt.imshow(m)
    #plt.show()
    # we generate the minimum bounding box for the extracted mask
    x,y,w,h = cv2.boundingRect(m.astype(np.uint8))
    
    if w < 10 or h < 10 or h > w:
        return 0
    if w >= p_shape[0]*0.8 or h >= p_shape[1]/4:
        return 0

    # we compute the score according to its shape and its size
    sc_shape = np.sum(m[y:y+h, x:x+w]) / (w*h)
    sc_size = (w*h) / (m.shape[0] * m.shape[1])
    #sc_textboxish = 1 - (8 - w / h) ** 2 / 64
    final_score = (sc_shape + 50*sc_size) / 2
        
    return final_score

def get_best_textbox_candidate(mask, original_mask):
    x,y,w,h = cv2.boundingRect(original_mask.astype(np.uint8))
    p_shape = (w,h)
    p_coords = (x,y)
    
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
        #print(f"{np.sum(biggest)} - {sc}")
        if sc > TH:
            #plt.imshow(biggest)
            #plt.show()
            mask = biggest
            best_sc = sc
            found = True
        else:
            mask_c -= biggest
            
    x, y, w, h = cv2.boundingRect(mask)
    M_W = 0.05
    M_H = 0.05
    ref = min(p_shape)
    x0,y0,x,y = (x - int(ref*M_W/2), y - int(ref*M_H/2), 
            (x+w) + int(ref*M_W/2), (y+h) + int(ref*M_H/2))
    return best_sc, [max(0,x0), max(0,y0), min(x, p_coords[0] + p_shape[0]), min(y, p_coords[1] + p_shape[1])]

def extract_textbox(orig_img, mask=None):
    masks = []
    shapes = []
    bboxes = []
    if mask is None:
        masks = [np.ones(orig_img.shape[:2])]
    else:
        masks = extract_paintings_from_mask(mask)
        
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

    #mask = generate_text_mask(orig_img.shape[0:2], bboxes)
    #plt.imshow(mask, cmap='gray')
    #plt.show()
    return bboxes


def generate_text_mask(shape, textboxes):
    if textboxes is None or len(textboxes) == 0:
        return np.zeros(shape).astype(np.uint8)
    
    mask = np.zeros(shape)
    for (xtl, ytl, xbr, ybr) in textboxes:
        pts = np.array(((xtl, ytl), (xtl, ybr), (xbr, ybr), (xbr, ytl)))
        cv2.fillConvexPoly(mask, pts, True)
    return mask.astype(np.uint8)

