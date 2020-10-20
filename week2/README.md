# Week 1

## Description of the task

* Create histogram image descriptors for both the museum images and the query sets.
* Implement similarity measures to compare the images.
* Implement a retrieval system to get the K top results.
* Evaluate the results using the MAP@k metric.
* Background removal of images from QS2.
* Evaluate the masks and the retrieval system for QS2.

## Evaluating query sets
Use `fastsearch.py` to get the map@K measure. To check its usage, run:

```$ python fastsearch.py -h```


Call structure (all parameters are compulsory):

```python fastsearch.py ["--museum_path MUSEUM_PATH] [--query_path QUERY_PATH] [--descriptor DESCRIPTOR] [--metric METRIC] [--bins BINS] [--map_k MAP_K]```

### Dataset-related parameters
1. `MUSEUM_PATH` is the path to reference museum dataset
2. `QUERY_PATH` is the path to query museum dataset

### Method-related parameters: 
1. `DESCRIPTOR` Use one of the available histogram retrieving methods:

        {gray_historam, rgb_histogram_1d, rgb_histogram_3d, hsv_histogram_1d, hsv_histogram_3d, lab_histogram_1d, lab_histogram_3d, ycrcb_histogram_1d, ycrcb_histogram_3d}
2. `METRIC` Use one of the available similarity measures:

        {cosine, manhattan, euclidean, intersect, kl_div, bhattacharyya, hellinger, chisqr, correl}
3. `BINS` Number of bins to use for the histograms
4. `MAP_K` Number of results (K) to use for MAP@K


## Background removal
Masks are generated and evaluated (if ground truth is available) by using the script `masks.py`, located in the `root/week1` folder. To check its usage, run:

```$ python masks.py -h```

Call structure: 

```python masks.py [--query QUERY] [--retriever RETRIEVER] [--output OUTPUT]```

Where all parameters are compulsory:
1. `QUERY` is the path to the query dataset
2.  `RETRIEVER` is the background removal strategy to use and must be one of the following options:

        {color_mono, color_multi_rgb, color_multi_hsv, color_multi_sv, color_multi_lab, color_multi_ycbcr, color_multi_xyz, edges}

3. `OUTPUT` refers to the folder where the generated masks will be stored (created if does not exist).