import numpy as np
from utils import generate_binary_mask
import os
import cv2
import pickle as pkl
import pandas as pd
import sys
sys.path.append(".")

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    #print(f"{actual} VS {predicted}")
    scores = []
    
    #print((actual, predicted))
    for (actual, predicted) in zip(actual, predicted):
        #print(f">{(actual, predicted)}")
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p == actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            scores.append(0.0)
        else:
            scores.append(score / min(1, k))
    return scores

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    apks = [apk(a, p, k) for a, p in zip(actual, predicted)]
    return np.mean([a for a_s in apks for a in a_s])


def evaluate_mask(mask: np.ndarray, mask_gt: np.ndarray):
    """
    Computes the precision, recall and F1-score of the mask compared to the ground truth mask (mask_gt).
    
    Parameters
    ----------
    mask : mask to evaluate
    mask_gt : ground truth mask
    
    Returns
    -------
    metrics : list containing the metrics (precision, recall, f1, tp, fp, fn, tn), where
                tp, fp, fn, tn correspond to true positives, false positives, false negatives and true negatives.
    """
    
    # we transform the masks to binary masks
    mask, mask_gt = generate_binary_mask(mask), generate_binary_mask(mask_gt)
    assert (mask.shape == mask_gt.shape)
    
    # flatten them to easily sum the results
    mask = mask.flatten()
    mask_gt = mask_gt.flatten()
    
    # we compute the number of False Positives (fp), False Negatives (fn), True Positives (tp) and True Negatives (tn)
    tp = int(np.sum(mask * mask_gt))
    fp = int(np.sum(mask * (1 - mask_gt)))
    fn = int(np.sum((1 - mask) * mask_gt))
    tn = int(np.sum((1 - mask) * (1 - mask_gt)))
    
    # we compute Precision, Recall and F1 metrics.
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1, tp, fp, fn, tn



if __name__ == '__main__':
    actual = [23]
    predicted = [45, 45, 31, 23 , 25]
    print(apk(actual, predicted))
