import argparse
import sys
import cv2
import glob, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
from distances import compute_distance
from histograms import extract_features
from functools import partial
import multiprocessing.dummy as mp

import random
from sklearn.cluster import KMeans

def get_image_path_list(data_path:str) -> List[Path]:
    return sorted(glob.glob(os.path.join(data_path,'*.jpg')))

def path2img(path:Path) -> np.ndarray:
    return np.array(Image.open(path))

museum_data = "/home/adityassrana/datatmp/Datasets/museum_dataset/processed/BBDD"
museum_list = get_image_path_list(museum_data)
print(f'number of images in reference dataset is {len(museum_list)}')

def batch_descriptors(museum_list:List[str],descriptor:str='rgb_histogram_3d', bins:int=8) -> List[int]:
    extract_features_func = partial(extract_features, descriptor=descriptor,bins=bins)
    def _extract_features_from_path(_path):
        return extract_features_func(path2img(_path))
    with mp.Pool(processes=20) as p:
        image_descriptors = p.map(_extract_features_from_path, [ref_path for ref_path in museum_list])
    return image_descriptors


def get_clusters(image_path_list, num_clusters:int=6, descriptor:str='rgb_histogram_3d',bins:int=8):
    features = batch_descriptors(image_path_list, descriptor=descriptor,bins=8)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(features)
    predictions = kmeans.predict(features)
    clusters = []
    for i in range(num_clusters):
        cluster_images = list(np.array(image_path_list)[np.where(predictions==i)])
        clusters.append(cluster_images)
    return clusters    

def plot_results(image_path_list: List[Path]) -> None:
    num_images = len(image_path_list)
    fig,axes = plt.subplots(int(num_images/6) + 1, 6, figsize=(24,24))
    for ax, path in zip(axes.flatten(), image_path_list):
        ax.imshow(np.array(Image.open(path)))
        ax.set_title(path.split('_')[-1])
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
def plot_clusters(clusters):
    for i, cluster in enumerate(clusters):
        print(f'Room {i}, length:{len(cluster)}')
        plot_results(cluster)
        plt.show()
    plt.show()

# to sample a mini-batch of images for experimentation
mini_museum_list = random.sample(museum_list,100)
#original BBDD data has 287 images
clusters = get_clusters(mini_museum_list, num_clusters = 10, descriptor = "dct_blocks", bins = 4)
plot_clusters(clusters)

