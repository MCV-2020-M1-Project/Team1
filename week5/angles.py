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
from pyclustering.cluster.kmedians import kmedians as kmed
import math
    
def compute_angle(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    #rads = np.arccos(np.dot(v1,v2))
    rads = np.arctan2(v2[1],v2[0]) - np.arctan2(v1[1],v1[0])
    grades = 180*rads/np.pi
    return grades


def rotate_point(img, point, angle):
    points = [point, ]
    (h, w) = img.shape[:2]
    origin = (w//2, h//2)
    M_inv = cv2.getRotationMatrix2D(origin, angle, 1.0)
    # add ones
    ones = np.ones(shape=(len(points), 1))

    points_ones = np.hstack([points, ones])

    # transform points
    transformed_points = M_inv.dot(points_ones.T).T
    return [int(round(p)) for p in transformed_points[0]]
    
    
def rotate_image(img, angle=0, mask=False):
    #angle = -1 * angle
    tmpImg = img.copy()
    (h, w) = img.shape[:2]
    origin = (w//2, h//2)
    mat = cv2.getRotationMatrix2D(origin, angle, 1.0)
    rotatedImg = cv2.warpAffine(
        tmpImg, mat, (w, h), flags=cv2.INTER_NEAREST if mask else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotatedImg

    
def extract_angle(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # First we find very long lines in the image
    dst = cv2.Canny(img, 50, 200)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 15, None, 10, img.shape[0] / 10)
    result_lines = np.zeros_like(img)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(result_lines, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv2.LINE_AA)

            
    # We get the contours, hoping we get as much as the frame contours as possible
    contours = cv2.findContours(result_lines.astype(np.uint8),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_copy = np.zeros_like(result_lines)
    cv2.drawContours(mask_copy, contours[1], -1, (255,255,255), thickness = 4)

    
    # We then extract the lines again, saving their angle respect the X axis.
    linesP = cv2.HoughLinesP(mask_copy, 1, np.pi / 180, 15, None, 50, 100)

    topaint2 = np.zeros_like(mask_copy)
    angles = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]

            v1 = (l[2] - l[0], l[3] - l[1])
            v1 = v1 / np.linalg.norm(v1)
            v2 = (1, 0)

            rads = np.arctan2(v2[1],v2[0]) - np.arctan2(v1[1],v1[0])
            grades = 180*rads/np.pi
            angles.append(grades)
        
    angles = np.array(angles)
    
    # Now, we get the window of 90º that maximizes the angles on the extremes, to discard FPs
    # (assuming all paintings are rectangles => 90º)
    max_, max_val = -90, 0
    current = -90
    eps = 5 # epsilon
    step = 0.1
    for i in range(int(90 / step)):
        current_count = np.where(((np.array(angles) > current - eps) & (np.array(angles) < current + eps)) | 
                                 ((np.array(angles) > current + 90 - eps) & (np.array(angles) < current + 90 + eps)))
        count = len(current_count[0])
        if count > max_val:
            max_val = count
            max_ = current
        current += step

        
    # We keep only those values on the extremes
    angles = [a for a in angles if (a > max_-eps and a < max_+eps) or 
                                    (a > max_-eps+90 and a < max_+eps+90)]
        
        
    # We find the median of the two clusters, initialized on the extreme values of the window
    kmedians_instance = kmed(np.array([(a,0) for a in angles]), [ [max_, 0.0], [max_+90,0.0] ]);
    kmedians_instance.process();
    kmedians_instance.get_clusters();
    
    # We return the minimum angle (in absolute value), which is the one we are looking for
    angles = [m[0] for m in kmedians_instance.get_medians()]
    angle = angles[np.argmin(np.abs(angles))]
    if angle < 0:
        return 180.0 + angle
    return angle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prints the angles for each image and evaluates the performance.")
    parser.add_argument('--query', help="Path to query folder.", type=str)
    parser.add_argument('--output', help="Path to output folder.", type=str)
    args = parser.parse_args()


    if not os.path.exists(args.query):
        print(f"[ERROR] Query folder '{args.query}' does not exist.")
        exit()


    print(f"Computing angles...")
    
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
    masks = [cv2.imread(os.path.join(args.query, f"{i:05d}.png"), 0) for i in range(db_count)]
    angles = [extract_angle(img) for img in tqdm(images)]
    
    
    acc = 0
    counter = 0
    for i in range(30):
        if i == 15:
            continue
        angle = angles[i]
        for j in range(len(frames[i])):
            gt = frames[i][j][0] if frames[i][j][0] < 90 else 180 - frames[i][j][0]
            test = angle if angle < 90 else 180 - angle
            print(f"[{i}]  {gt:.2f} / {test:.2f}")
            acc += np.abs(gt - test)
            counter += 1
    acc /= counter
    
    print(f"Done! Mean angle distance: '{acc}'.")
    
    if args.output:
        print(f"Saving rotated images to '{args.output}'...")
        if not os.path.exists(args.output):
            print(f"Creating output folder '{args.output}'...")
            os.mkdir(args.output)
            
        for i in range(db_count):
            angle = angles[i]
            if angle > 90:
                angle = angle - 180.0
            cv2.imwrite(os.path.join(args.output, f"{i:05d}.jpg"), rotate_image(images[i], -angle))
            cv2.imwrite(os.path.join(args.output, f"{i:05d}.png"), rotate_image(masks[i], -angle))
        
        
    print(f"Successfully processed {db_count} images in {time.time() - t0:.2f}s!")


