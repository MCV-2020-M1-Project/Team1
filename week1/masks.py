
import cv2
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import time
from evaluation import evaluate_mask

def evaluate_mask_retriever(query_db_path: str, retriever: str, *, output: str = None):
    """
    Evaluates the given mask retriever against all the images in the query dataset.
    
    Parameters
    ----------
    query_db_path : path where que images of the query dataset are located
    retriever : mask method to evaluate
    
    Returns
    -------
    pr: precision
    rec: recall
    f1: F1-score
    results: pandas dataframe that contains image-wise metrics.
    """
    
    if not os.path.exists(query_db_path):
        print("[ERROR] Query dataset path is wrong.")
        exit()
    if output is not None and not os.path.exists(output):
        print("[ERROR] Output folder does not exist.")
        exit()
        
    # We load the dataset metadata
    with open(os.path.join(query_db_path, "frames.pkl"), 'rb') as file:
        frames = pkl.load(file)
    db_count = len(frames)
    #print(f"There are {db_count} images in the query dataset.")

    # We load the images and their associated masks
    images = [cv2.imread(os.path.join(query_db_path, f"{i:05d}.jpg")) for i in range(db_count)]
    masks = [cv2.imread(os.path.join(query_db_path, f"{i:05d}.png")) for i in range(db_count)]
    
    # We generate the masks and compare it with the GT mask, computing several metrics
    generated_masks = [extract_mask(images[i], retriever) for i in range(db_count)]
    if output is not None:
        # we save all generated masks
        for i in range(db_count):
            cv2.imwrite(os.path.join(output, f"{i:05d}.png"), generated_masks[i] * 255)

    print(f"[INFO] Masks successfully stored in '{output}'")
    try:
        data = [(i, ) + evaluate_mask(generated_masks[i], masks[i]) for i in range(db_count)]
        results = pd.DataFrame(data=data, columns=("index", "precision", "recall", "f1", "tp", "fp", "fn", "tn"))
        # We print the average metrics for the whole query dataset.
        pr, rec, f1 = results["precision"].mean(), results["recall"].mean(), results["f1"].mean()
        return pr, rec, f1, results
    except:
        print("[INFO] GT Masks not found => evaluation not performed.")
        return None
    


def __extract_biggest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Extracts the biggest connected component from a mask (0 and 1's).

    Args:
        img: 2D array of type np.float32 representing the mask
    
    Returns : 2D array, mask with 1 in the biggest component and 0 outside
    """
    # extract all connected components
    num_labels, labels_im = cv2.connectedComponents(mask.astype(np.uint8))
    
    # we find and return only the biggest one
    max_val, max_idx = 0, -1
    for i in range(1, num_labels):
        area = np.sum(labels_im == i)
        if area > max_val:
            max_val = area
            max_idx = i
            
    return (labels_im == max_idx).astype(float)


def extract_mask_based_on_color_graph_mono(img: np.ndarray, *, use_diagonals: bool = False) -> np.ndarray:
    """
    Extracts the mask for the painting by flood filling the background starting from the
    image boundaries. Then it returns the biggest connected component.

    Args:
        img: 2D array of type np.float32 representing the image
    
    Returns : 2D array, mask with 1 in the foreground and 0 in the background
    """
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert len(img.shape) == 2
    
    # we normalize the 2D input image
    img = 255 * ((img - np.min(img)) / (np.max(img) - np.min(img)))
    EPSILON = 5
    
    # we initialize the mask to all 1.
    mask = np.zeros_like(img) + 1
    # matrix of visited pixels
    visited = np.zeros_like(img)
    w, h = img.shape
    
    # pixels where we start the background removal from (edge of the image)
    pixels = [(x,0) for x in range(w)] + [(0,y) for y in range(h)]
    for (x0, y0) in pixels:
        mask[x0,y0] = 0
        visited[x0,y0] = 1
        
    # Flood fill algorithm
    while len(pixels) > 0:
        # Starting point
        x0, y0 = pixels.pop()
        # Directions where we will go to
        steps = ((0,-1), (0,1), (-1,0), (1,0))
        if use_diagonals:
            steps += ((1,1), (1,-1), (-1,1), (-1,-1))
       
        for (hx, hy) in steps:
            x, y = (x0+hx, y0+hy)
            # We check if the step is valid:
            # 1) within image boundaries
            # 2) not visited yet
            # 3) color intensity change lower than EPSILON
            if x >= 0 and y >= 0 and x < w and y < h and visited[x, y] == 0 and abs(img[x,y] - img[x0,y0]) <= EPSILON:
                # We add it as background
                mask[x,y] = 0
                visited[x,y] = 1
                # It will be an initial point for the flood algorithm later on
                pixels.append((x,y))
        
  
    return __extract_biggest_connected_component(mask).astype(float)


def extract_mask_based_on_color_graph_multi(img, *, mode="hsv"):
    """
    Applies 'extract_mask_based_on_color_graph_mono' algorithm to different channels and selects
    the best mask according to:
    - Amount of overlap with the minimum bounding box where is contained. => the CLOSER, the better,
        as paintings are supposed to be rectangular, and we suppose images are not extremely rotated.
    - Size of such bounding box => the BIGGER, the better (to avoid missing the frame)

    Args:
        img: 2D array of type np.float32 representing the image
        mode: colorspace to use∫
    
    Returns : 2D array, mask with 1 in the foreground and 0 in the background
    """
    
    #print(f"[COLOR_MULTI] mode={mode}")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    if mode == "hsv":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif mode == "lab":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif mode == "xyz":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    elif mode == "ycbcr":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif mode == "sv": # like hsv but only with saturation and lightness
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1:]
    elif mode != "bgr":
        print("[ERROR] Unknown mode.")
        return None
    
    
    final_mask = None
    max_score = 0
    # we apply the algorithm for each channel
    for i in range(img.shape[2]):
        mask = extract_mask(img[:,:,i], "color_mono")
        
        # we generate the minimum bounding box for the extracted mask
        x,y,w,h = cv2.boundingRect(mask.astype(np.uint8))
        
        # we compute the score according to its shape and its size
        sc_shape = np.sum(mask[y:y+h, x:x+w]) / (w*h)
        sc_size = (w*h) / (img.shape[0] * img.shape[1])
        score = sc_shape + sc_size
        
        # we update if it is the one with the best score
        if score > max_score:
            max_score = score
            final_mask = mask
            
    return final_mask


def extract_mask_based_on_edges_v1(img:np.ndarray, *, min_th: float = 100, max_th: float = 200) -> np.ndarray:
    """
    Extracts the mask for the painting by extracting the minimum rectangular
    bounding box containing the biggest contour in the image.

    Args:
        img: 2D or 3D array of type np.float32 representing the image
    
    Returns : 2D array, mask with 1 in the foreground and 0 in the background
    """
    #print(f"[EDGES] min_th={min_th} and max_th={max_th}")
    img = img.copy()
        
    # opencv canny method works with both grayscale and color
    edges = cv2.Canny(img, min_th, max_th)
    
    # we get all the contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # we select the contour that can be enclosed inside the biggest rectangle
    # TODO work with rotated rectangles instead in a V2
    max_area = 0
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        xc,yc,wc,hc = cv2.boundingRect(cnt)
        if wc * hc > max_area:
            max_area = wc * hc
            x,y,w,h = xc,yc,wc,hc
    
    # we generate and return the mask
    mask = np.zeros(img.shape[:2]) + 1
    mask[:, 0:x+1], mask[0:y+1, :], mask[:, x+w-1:], mask[y+h-1:, :] = 0, 0, 0, 0
    return mask



BG_RETRIEVERS = {
    "color_mono": (extract_mask_based_on_color_graph_mono, {}),
    "color_multi_rgb": (extract_mask_based_on_color_graph_multi, {"mode": "bgr",}),
    "color_multi_hsv": (extract_mask_based_on_color_graph_multi, {"mode": "hsv",}),
    "color_multi_sv": (extract_mask_based_on_color_graph_multi, {"mode": "sv",}),
    "color_multi_lab": (extract_mask_based_on_color_graph_multi, {"mode": "lab",}),
    "color_multi_ycbcr": (extract_mask_based_on_color_graph_multi, {"mode": "ycbcr",}),
    "color_multi_xyz": (extract_mask_based_on_color_graph_multi, {"mode": "xyz",}),
    "edges": (extract_mask_based_on_edges_v1, {"min_th": 100, "max_th": 200,}),
}


def extract_mask(image:np.ndarray, retriever:str) -> np.ndarray:
    """
    Extract features from image based on retriever of choice

    Args:
        image: (H x W x C) 2D (grayscale) or 3D (RGB) image array
        retriever: method to use to generate mask
        
    Returns: 
        2D array, mask with 1 in the foreground and 0 in the background
    """
    return BG_RETRIEVERS[retriever][0](image, **BG_RETRIEVERS[retriever][1])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates, evaluates and stores (optional, see --output) masks generated from the given query dataset. ")
    parser.add_argument('--query', help="Path to query dataset.", type=str)
    parser.add_argument('--retriever', help="Mask retriever method to use.", type=str, choices=list(BG_RETRIEVERS.keys()),)
    parser.add_argument('--output', help="Path to store generated masks. Results are not saved if unspecified.", type=str,)
    args = parser.parse_args()


    if not os.path.exists(args.query):
        print(f"[ERROR] Query path '{args.query}' does not exists.")
        exit()
    if not os.path.exists(args.output):
        print("Creating output folder...")
        os.mkdir(args.output)


    print(f"Evaluating with mask retriever >{args.retriever}<...")
    t0 = time.time()
    result = evaluate_mask_retriever(args.query, args.retriever, output=args.output)
    t = time.time()
    if result is not None:
        pr, rec, f1, results = result

        print(f"[{t-t0}s] Precision: {pr:.3f}, Recall: {rec:.3f}, F1-Score: {f1:.3f}")

        to_save_path = os.path.join(args.output, "results.csv")
        results.to_csv(to_save_path, index=False)
        print(f"Results successfully saved in '{to_save_path}'")
