# Week 3 API Usage

## Content Based Image Retrieval

Use `fastsearch.py` to get the map@K measure. To check its usage, run:

```$ python fastsearch.py -h```

````
$ python fastsearch.py -h
usage: fastsearch.py [-h] [--museum_path MUSEUM_PATH]
                     [--query_path QUERY_PATH] [--pickle] [--plot]
                     [--use_masks] [--filter_text] [--use_color] [--use_text]
                     [--use_texture] [--masks_multiple]
                     [--color_weight COLOR_WEIGHT]
                     [--color_descriptor COLOR_DESCRIPTOR]
                     [--color_metric COLOR_METRIC] [--color_bins COLOR_BINS]
                     [--text_weight TEXT_WEIGHT] [--text_reader TEXT_READER]
                     [--text_metric TEXT_METRIC]
                     [--texture_weight TEXTURE_WEIGHT]
                     [--texture_descriptor TEXTURE_DESCRIPTOR]
                     [--texture_metric TEXTURE_METRIC]
                     [--texture_bins TEXTURE_BINS] [--map_k MAP_K]

Content Based Image Retrieval

optional arguments:
  -h, --help            show this help message and exit
  --museum_path MUSEUM_PATH, -r MUSEUM_PATH
                        path to reference museum dataset. Example input:
                        'data/BBDD'
  --query_path QUERY_PATH, -q QUERY_PATH
                        path to query museum dataset. Example input:
                        'data/qsd1_w1'
  --pickle, -p          Generate pickle file with results
  --plot, -v            show retrieval results for a random query image
  --use_masks           whether to use masks for histogram generation or not.
                        Using masks helps us improve our features by extract
                        the painting(foreground) from the background and also
                        removing any text present on the painting
  --filter_text         whether textboxes will be loaded from 'text_boxes.pkl'
                        file and removed from the color and texture feature
                        extractions.
  --use_color           whether color matching will be used to do the
                        retrieval.
  --use_text            whether author matching will be used to do the
                        retrieval.
  --use_texture         whether texture matching will be used to do the
                        retrieval.
  --masks_multiple      whether several paintings can appear in the mask or
                        not (with a current maximum of 2).
  --color_weight COLOR_WEIGHT, -cow COLOR_WEIGHT
                        weight for the color matching
  --color_descriptor COLOR_DESCRIPTOR
                        descriptor for extracting features from image.
                        DESCRIPTORS AVAILABLE: 1D and 3D Histograms -
                        gray_historam, rgb_histogram_1d, rgb_histogram_3d,
                        hsv_histogram_1d, hsv_histogram_3d, lab_histogram_1d,
                        lab_histogram_3d, ycrcb_histogram_1d,
                        ycrcb_histogram_3d. Block and Pyramidal Histograms -
                        lab_histogram_3d_pyramid and more.
                        lab_histogram_3d_blocks gives us the best results.
  --color_metric COLOR_METRIC
                        similarity measure to compare images. METRICS
                        AVAILABLE: cosine, manhattan, euclidean, intersect,
                        kl_div, js_div bhattacharyya, hellinger, chisqr,
                        correl. hellinger and js_div give the best results.
  --color_bins COLOR_BINS
                        number of bins to use for histograms
  --text_weight TEXT_WEIGHT, -txw TEXT_WEIGHT
                        weight for the text matching
  --text_reader TEXT_READER
                        OCR algorithm used to extract the text from inside the
                        textbox. READERS AVAILABLE: tesseract
  --text_metric TEXT_METRIC
                        Metric used to compare extracted text with paintings
                        text from the database. METRICS AVAILABLE:
                        ratcliff_obershelp, levenshtein, cosine
  --texture_weight TEXTURE_WEIGHT, -tuw TEXTURE_WEIGHT
                        weight for the color matching
  --texture_descriptor TEXTURE_DESCRIPTOR
                        descriptor for extracting textures from image.
                        DESCRIPTORS AVAILABLE: 1D and 3D Histograms -
                        gray_historam, rgb_histogram_1d, rgb_histogram_3d,
                        hsv_histogram_1d, hsv_histogram_3d, lab_histogram_1d,
                        lab_histogram_3d, ycrcb_histogram_1d,
                        ycrcb_histogram_3d. Block and Pyramidal Histograms -
                        lab_histogram_3d_pyramid and more. lab_histogram_3d
                        gives us the best results.
  --texture_metric TEXTURE_METRIC
                        textures similarity measure to compare images. METRICS
                        AVAILABLE: cosine, manhattan, euclidean, intersect,
                        kl_div, js_div bhattacharyya, hellinger, chisqr,
                        correl. hellinger and js_div give the best results.
  --texture_bins TEXTURE_BINS
                        number of bins to use for textures histograms
  --map_k MAP_K, -k MAP_K
                        Mean average precision of top-K results
````


## Background removal
Masks are generated and evaluated (if ground truth is available) by using the script `masks.py`. To check its usage, run:

```$ python masks.py -h```

````
$ python masks.py -h
usage: masks.py [-h] [--query QUERY]
                [--retriever {color_mono,color_multi_rgb,color_multi_hsv,color_multi_sv,color_multi_lab,color_multi_ycbcr,color_multi_xyz,edges}]
                [--output OUTPUT]

Generates, evaluates and stores (optional, see --output) masks generated from
the given query dataset.

optional arguments:
  -h, --help            show this help message and exit
  --query QUERY         Path to query dataset.
  --retriever {color_mono,color_multi_rgb,color_multi_hsv,color_multi_sv,color_multi_lab,color_multi_ycbcr,color_multi_xyz,edges}
                        Mask retriever method to use. Options Available:
                        color_mono, color_multi_rgb, color_multi_hsv,
                        color_multi_sv, color_multi_lab, color_multi_ycbcr,
                        color_multi_xyz, edges
  --output OUTPUT       Path to folder where generated masks will be stored.
                        Results are not saved if unspecified.
````
## Visualizing Results

Use command line argument "-v" to plot the top results for a random query image.

````python fastsearch.py -v````

<br>
<img src="images/plot_results.png" height=600>